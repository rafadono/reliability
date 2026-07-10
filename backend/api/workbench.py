import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import state
from src.reliability_analysis.analysis.models import ReliabilityFitter, KijimaFitter
from src.reliability_analysis.core.data_processing import DataProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

# Schema persistence path
DATA_DIR = Path(__file__).parent.parent / "data"
PIPELINES_FILE = DATA_DIR / "workbench_pipelines.json"
DATA_DIR.mkdir(exist_ok=True)

class NodeSchema(BaseModel):
    id: str
    type: str
    data: Dict[str, Any]
    x: float
    y: float

class EdgeSchema(BaseModel):
    id: str
    source: str
    target: str

class PipelineExecuteRequest(BaseModel):
    nodes: List[NodeSchema]
    edges: List[EdgeSchema]

class PipelineSaveRequest(BaseModel):
    name: str
    nodes: List[NodeSchema]
    edges: List[EdgeSchema]


def topological_sort(nodes: List[NodeSchema], edges: List[EdgeSchema]) -> List[NodeSchema]:
    """Sort nodes in execution order using Kahn's algorithm."""
    adj = {n.id: [] for n in nodes}
    in_degree = {n.id: 0 for n in nodes}
    node_map = {n.id: n for n in nodes}
    
    for e in edges:
        src, tgt = e.source, e.target
        if src in adj and tgt in in_degree:
            adj[src].append(tgt)
            in_degree[tgt] += 1
            
    # Nodes with 0 in-degree
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    order = []
    
    while queue:
        u = queue.pop(0)
        order.append(node_map[u])
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    if len(order) < len(nodes):
        raise ValueError("Se detectó una relación cíclica en el flujo del Workbench.")
        
    return order


def execute_datasource_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    if state.current_data is None:
        raise ValueError("No hay datos cargados en el sistema.")
    
    df = state.current_data.copy()
    
    # Return basic metadata to the UI
    available_equipment = df["Equipment"].dropna().unique().tolist() if "Equipment" in df.columns else []
    
    return {
        "df": df,
        "ui_data": {
            "rows": len(df),
            "columns": list(df.columns),
            "available_equipment": available_equipment
        }
    }


def execute_filter_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    df = inputs.get("df")
    if df is None:
        raise ValueError("Nodo de filtrado no recibe datos de entrada.")
        
    params = node.data
    equipment = params.get("equipment", "")
    failure_type = params.get("type", "")
    mdf = params.get("mdf", "")
    censored = params.get("censored", None)
    
    filtered_df = df.copy()
    if equipment:
        filtered_df = filtered_df[filtered_df["Equipment"] == equipment]
        
    if failure_type:
        if isinstance(failure_type, list):
            if len(failure_type) > 0:
                filtered_df = filtered_df[filtered_df["Type"].isin(failure_type)]
        else:
            filtered_df = filtered_df[filtered_df["Type"] == failure_type]
            
    if mdf:
        if isinstance(mdf, list):
            if len(mdf) > 0:
                filtered_df = filtered_df[filtered_df["mdf"].isin(mdf)]
        else:
            filtered_df = filtered_df[filtered_df["mdf"] == mdf]
            
    if censored is not None and censored != "" and censored != "all":
        filtered_df = filtered_df[filtered_df["Censored"] == int(censored)]
        
    available_types = filtered_df["Type"].dropna().unique().tolist() if "Type" in filtered_df.columns else []
    available_mdfs = filtered_df["mdf"].dropna().unique().tolist() if "mdf" in filtered_df.columns else []
    
    return {
        "df": filtered_df,
        "ui_data": {
            "rows": len(filtered_df),
            "equipment": equipment,
            "type": failure_type,
            "mdf": mdf,
            "available_types": available_types,
            "available_mdfs": available_mdfs
        }
    }


def execute_weibull_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    df = inputs.get("df")
    if df is None:
        raise ValueError("El nodo Weibull requiere un conjunto de datos filtrado.")
        
    if df.empty or len(df) < 2:
        return {"error": "Insuficientes datos para ajustar Weibull (mínimo 2 registros)."}
        
    try:
        fitter = ReliabilityFitter(df)
        res = fitter.fit_weibull()
        
        if not res:
            return {"error": "No se pudo ajustar la distribución Weibull."}
        if "error" in res:
            return {"error": res["error"]}
            
        beta = res.get("beta")
        eta = res.get("eta")
        mtbf = res.get("mtbf")
        
        mttr_val = 0.0
        if "TTX" in df.columns:
            failures_df = df[df["Censored"] == 0]
            if not failures_df.empty:
                mttr_val = float(failures_df["TTX"].mean())
            else:
                mttr_val = float(df["TTX"].mean())
                
        return {
            "beta": round(beta, 4) if beta is not None else None,
            "eta": round(eta, 4) if eta is not None else None,
            "mtbf": round(mtbf, 2) if mtbf is not None else None,
            "mttr": round(mttr_val, 2),
            "aic": round(res.get("aic"), 2) if res.get("aic") is not None else None,
            "bic": round(res.get("bic"), 2) if res.get("bic") is not None else None,
            "ks_p_value": round(res.get("p_value"), 4) if res.get("p_value") is not None else None
        }
    except Exception as e:
        return {"error": f"Fallo al ajustar Weibull: {str(e)}"}


def execute_kijima_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    df = inputs.get("df")
    if df is None:
        raise ValueError("El nodo Kijima requiere un conjunto de datos filtrado.")
        
    if df.empty or len(df) < 3:
        return {"error": "Insuficientes datos para ajustar Kijima (mínimo 3 registros)."}
        
    model_type = int(node.data.get("model_type", 1))
    
    try:
        column = "TBX" if "TBX" in df.columns else "Days"
        if column not in df.columns:
            column = "TTX" if "TTX" in df.columns else "Days"
            
        df_clean = df[df[column] > 0].copy()
        
        fitter = KijimaFitter()
        res = fitter.fit(
            dataframe=df_clean,
            column=column,
            censored_types=[],
            models=[model_type]
        )
        
        if isinstance(res, list):
            res = res[0]
            
        return {
            "model_name": res["model_name"],
            "beta": round(res["beta"], 4),
            "eta": round(res["eta"], 4),
            "ar": round(res["ar"], 4) if res.get("ar") is not None else None,
            "ap": round(res["ap"], 4) if res.get("ap") is not None else None,
            "r2": round(res.get("ks_stat", 0.0), 4)
        }
    except Exception as e:
        logger.error(f"Error in Kijima node: {str(e)}")
        return {"error": f"Fallo al ajustar Kijima: {str(e)}"}


def execute_fmeca_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    # Custom interactive table node, performs RPN calculations
    records = node.data.get("records", [])
    processed_records = []
    
    for r in records:
        sev = int(r.get("severity", 5))
        occ = int(r.get("occurrence", 5))
        det = int(r.get("detection", 5))
        rpn = sev * occ * det
        
        if rpn < 50:
            cat = "Bajo"
        elif rpn < 150:
            cat = "Medio"
        elif rpn < 300:
            cat = "Alto"
        else:
            cat = "Crítico"
            
        processed_records.append({
            "component": r.get("component", ""),
            "mode": r.get("mode", ""),
            "effect": r.get("effect", ""),
            "severity": sev,
            "occurrence": occ,
            "detection": det,
            "rpn": rpn,
            "category": cat,
            "action": r.get("action", "")
        })
        
    return {
        "records": processed_records
    }


def execute_ram_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    df = inputs.get("df")
    if df is None:
        raise ValueError("El simulador RAM requiere un conjunto de datos.")
        
    if df.empty:
        raise ValueError("Dataset vacío en simulador RAM.")
        
    prev_eff = float(node.data.get("preventive_efficiency", 0.8))
    log_delay = float(node.data.get("logistics_delay", 4.0))
    
    # Calculate downtime and failures
    correctives = df[df["Censored"] == 0]
    n_failures = len(correctives)
    d_real = float(df["TTX"].sum()) if "TTX" in df.columns else 0.0
    
    # Mathematical Model (ISO 20815)
    d_logistics = log_delay * n_failures
    d_simulated = (d_real + d_logistics) * (1 - (prev_eff * 0.40))
    d_simulated = max(10.0, min(8660.0, d_simulated))
    
    uptime = 8760.0 - d_simulated
    availability = (uptime / 8760.0) * 100.0
    prod_assurance = availability * 0.985
    
    return {
        "availability": round(availability, 2),
        "production_assurance": round(prod_assurance, 2),
        "uptime_hours": round(uptime, 1),
        "downtime_hours": round(d_simulated, 1),
        "failures_count": n_failures
    }


def execute_pareto_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    df = inputs.get("df")
    if df is None:
        raise ValueError("El análisis de Pareto requiere un conjunto de datos.")
    
    from src.reliability_analysis.analysis.pareto import ParetoAnalyzer
    group_by = node.data.get("group_by", "Equipment")
    if group_by.lower() in ("equipo", "equipment"):
        result = ParetoAnalyzer.analyze_by_equipment(df)
    elif group_by.lower() in ("tipo", "type"):
        result = ParetoAnalyzer.analyze_by_type(df)
    else:
        result = ParetoAnalyzer.analyze_by_failure_mode(df)
        
    vital, trivial, stats = ParetoAnalyzer.get_80_20_split(result)
    
    return {
        "group_by": group_by,
        "vital_few": vital[:5],
        "stats": stats
    }


def execute_jackknife_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    df = inputs.get("df")
    if df is None:
        raise ValueError("El análisis Jackknife requiere un conjunto de datos.")
        
    group_col = node.data.get("compare_by", "Equipment")
    if group_col not in df.columns:
        group_col = "Equipment" if "Equipment" in df.columns else df.columns[0]
        
    stats = (
        df.groupby(group_col)
        .agg(
            failures=(group_col, "count"),
            total_downtime=("TTX", "sum"),
            avg_downtime=("TTX", "mean"),
        )
        .reset_index()
    )
    
    total_failures = float(stats["failures"].sum())
    avg_failures = float(stats["failures"].mean()) if not stats.empty else 0
    avg_total = float(stats["total_downtime"].mean()) if not stats.empty else 0
    
    critical_items = []
    chronic_items = []
    acute_items = []
    
    for _, row in stats.iterrows():
        name = str(row[group_col])
        x = float(row["failures"])
        y = float(row["total_downtime"])
        
        if x > avg_failures and y > avg_total:
            critical_items.append(name)
        elif x <= avg_failures and y > avg_total:
            acute_items.append(name)
        elif x > avg_failures and y <= avg_total:
            chronic_items.append(name)
            
    return {
        "critical_count": len(critical_items),
        "chronic_count": len(chronic_items),
        "acute_count": len(acute_items),
        "critical_list": critical_items[:5],
        "chronic_list": chronic_items[:5],
        "acute_list": acute_items[:5]
    }


def execute_trend_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    df = inputs.get("df")
    if df is None:
        raise ValueError("El análisis de tendencia requiere un conjunto de datos.")
        
    failures = len(df)
    downtime = float(df["TTX"].sum()) if "TTX" in df.columns else 0.0
    uptime = float(df["TBX"].sum()) if "TBX" in df.columns else 0.0
    failures_mtbf = int((df["TBX"] > 0).sum()) if "TBX" in df.columns else failures
    mtbf = float(uptime / failures_mtbf) if failures_mtbf > 0 else 0.0
    mttr = float(downtime / failures) if failures > 0 else 0.0
    total_time = uptime + downtime
    availability = float((uptime / total_time) * 100.0) if total_time > 0.0 else 0.0
    
    return {
        "failures": failures,
        "mtbf": round(mtbf, 2),
        "mttr": round(mttr, 2),
        "availability": round(availability, 2)
    }


@router.post("/workbench/execute", tags=["Workbench"])
async def execute_pipeline(req: PipelineExecuteRequest) -> Dict[str, Any]:
    """Sorts and executes the analytical pipeline nodes sequentially."""
    try:
        execution_order = topological_sort(req.nodes, req.edges)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
        
    # Build connections graph mapping source -> target
    connections = {}
    for e in req.edges:
        connections[e.target] = e.source
        
    results = {}
    context = {} # Stores pandas DataFrames (df) passed between nodes
    
    for node in execution_order:
        node_id = node.id
        node_type = node.type
        
        # Resolve inputs
        parent_id = connections.get(node_id)
        node_inputs = {}
        if parent_id and parent_id in context:
            node_inputs = context[parent_id]
            
        try:
            if node_type == "dataSource":
                res = execute_datasource_node(node, node_inputs)
                context[node_id] = {"df": res["df"]}
                results[node_id] = {"status": "success", "output": res["ui_data"]}
                
            elif node_type == "filter":
                res = execute_filter_node(node, node_inputs)
                context[node_id] = {"df": res["df"]}
                results[node_id] = {"status": "success", "output": res["ui_data"]}
                
            elif node_type == "weibull":
                res = execute_weibull_node(node, node_inputs)
                results[node_id] = {"status": "success", "output": res}
                
            elif node_type == "kijima":
                res = execute_kijima_node(node, node_inputs)
                results[node_id] = {"status": "success", "output": res}
                
            elif node_type == "fmeca":
                res = execute_fmeca_node(node, node_inputs)
                results[node_id] = {"status": "success", "output": res}
                
            elif node_type == "ramSimulator":
                res = execute_ram_node(node, node_inputs)
                results[node_id] = {"status": "success", "output": res}
                
            elif node_type == "pareto":
                res = execute_pareto_node(node, node_inputs)
                results[node_id] = {"status": "success", "output": res}
                
            elif node_type == "jackknife":
                res = execute_jackknife_node(node, node_inputs)
                results[node_id] = {"status": "success", "output": res}
                
            elif node_type == "trend":
                res = execute_trend_node(node, node_inputs)
                results[node_id] = {"status": "success", "output": res}
                
            else:
                results[node_id] = {"status": "error", "error": f"Tipo de nodo '{node_type}' no soportado."}
                
        except Exception as node_err:
            logger.error(f"Error executing node {node_id} ({node_type}): {str(node_err)}")
            results[node_id] = {
                "status": "error",
                "error": str(node_err)
            }
            
    return {
        "status": "success",
        "results": results
    }


@router.post("/workbench/save", tags=["Workbench"])
async def save_pipeline(req: PipelineSaveRequest) -> Dict[str, Any]:
    """Persists pipeline configurations to a local JSON file."""
    try:
        pipelines = {}
        if PIPELINES_FILE.exists():
            with open(PIPELINES_FILE, "r", encoding="utf-8") as f:
                pipelines = json.load(f)
                
        pipelines[req.name] = {
            "name": req.name,
            "nodes": [n.model_dump() for n in req.nodes],
            "edges": [e.model_dump() for e in req.edges]
        }
        
        with open(PIPELINES_FILE, "w", encoding="utf-8") as f:
            json.dump(pipelines, f, indent=2, ensure_ascii=False)
            
        return {"status": "success", "message": f"Pipeline '{req.name}' guardado correctamente."}
    except Exception as e:
        logger.error(f"Error saving pipeline: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/workbench/list", tags=["Workbench"])
async def list_pipelines() -> Dict[str, Any]:
    """Lists names of all saved pipelines."""
    try:
        pipelines = {}
        if PIPELINES_FILE.exists():
            with open(PIPELINES_FILE, "r", encoding="utf-8") as f:
                pipelines = json.load(f)
        return {"status": "success", "pipelines": list(pipelines.keys())}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/workbench/load/{name}", tags=["Workbench"])
async def load_pipeline(name: str) -> Dict[str, Any]:
    """Loads a specific pipeline configuration by name."""
    try:
        pipelines = {}
        if PIPELINES_FILE.exists():
            with open(PIPELINES_FILE, "r", encoding="utf-8") as f:
                pipelines = json.load(f)
                
        if name not in pipelines:
            raise HTTPException(status_code=404, detail=f"Pipeline '{name}' no encontrado.")
            
        return {"status": "success", "pipeline": pipelines[name]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading pipeline: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
