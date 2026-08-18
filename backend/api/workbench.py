import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import state
from src.reliability_analysis.analysis.models import ReliabilityFitter, KijimaFitter
from api.analysis import (
    compute_criticality,
    compute_bad_actors,
    compute_event_plot,
    compute_ram_simulation,
)
from services.llm import LlmService

logger = logging.getLogger(__name__)
router = APIRouter()


def resolve_node_equipment(node: "NodeSchema", df) -> Optional[str]:
    """
    Resolves which equipment an RCM/RCA node should analyze: an explicit
    node.data.equipment set in the inspector takes priority; otherwise, if the
    upstream (filtered) data narrows down to a single equipment, use that.
    """
    explicit = (node.data or {}).get("equipment")
    if explicit:
        return explicit
    if df is not None and "Equipment" in df.columns:
        uniques = df["Equipment"].dropna().unique().tolist()
        if len(uniques) == 1:
            return str(uniques[0])
    return None


# Schema persistence path
DATA_DIR = Path(__file__).parent.parent / "data"
PIPELINES_FILE = DATA_DIR / "workbench_pipelines.json"
DATA_DIR.mkdir(exist_ok=True)


class NodeSchema(BaseModel):
    id: str
    type: str
    data: Dict[str, Any] = {}
    x: Optional[float] = 0.0
    y: Optional[float] = 0.0


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


def topological_sort(
    nodes: List[NodeSchema], edges: List[EdgeSchema]
) -> List[NodeSchema]:
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
    available_equipment = (
        df["Equipment"].dropna().unique().tolist() if "Equipment" in df.columns else []
    )

    return {
        "df": df,
        "ui_data": {
            "rows": len(df),
            "columns": list(df.columns),
            "available_equipment": available_equipment,
        },
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
    censored_types = params.get("censored_types", [])
    censored_mdfs = params.get("censored_mdfs", [])

    filtered_df = df.copy()
    if equipment:
        if (
            "Equipment" in filtered_df.columns
            and (filtered_df["Equipment"] == equipment).any()
        ):
            filtered_df = filtered_df[filtered_df["Equipment"] == equipment]
        else:
            # Clear stale filter if equipment does not exist in new dataset
            equipment = ""
            node.data["equipment"] = ""
            node.data["type"] = []
            node.data["mdf"] = []
            node.data["censored_types"] = []
            node.data["censored_mdfs"] = []

    if failure_type:
        if isinstance(failure_type, list):
            if len(failure_type) > 0 and "Type" in filtered_df.columns:
                valid_types = [
                    t for t in failure_type if (filtered_df["Type"] == t).any()
                ]
                if valid_types:
                    filtered_df = filtered_df[filtered_df["Type"].isin(valid_types)]
                else:
                    node.data["type"] = []
        elif (
            "Type" in filtered_df.columns
            and (filtered_df["Type"] == failure_type).any()
        ):
            filtered_df = filtered_df[filtered_df["Type"] == failure_type]

    if mdf:
        if isinstance(mdf, list):
            if len(mdf) > 0 and "mdf" in filtered_df.columns:
                valid_mdfs = [m for m in mdf if (filtered_df["mdf"] == m).any()]
                if valid_mdfs:
                    filtered_df = filtered_df[filtered_df["mdf"].isin(valid_mdfs)]
                else:
                    node.data["mdf"] = []
        elif "mdf" in filtered_df.columns and (filtered_df["mdf"] == mdf).any():
            filtered_df = filtered_df[filtered_df["mdf"] == mdf]

    # Explicitly flag Censored = 1 for matching censored_types or censored_mdfs
    if censored_types or censored_mdfs:
        import pandas as pd

        if "Censored" not in filtered_df.columns:
            filtered_df["Censored"] = 0

        mask_censored = pd.Series(False, index=filtered_df.index)
        if censored_types and len(censored_types) > 0 and "Type" in filtered_df.columns:
            mask_censored = mask_censored | filtered_df["Type"].isin(censored_types)
        if censored_mdfs and len(censored_mdfs) > 0 and "mdf" in filtered_df.columns:
            mask_censored = mask_censored | filtered_df["mdf"].isin(censored_mdfs)

        filtered_df.loc[mask_censored, "Censored"] = 1
        filtered_df.loc[~mask_censored, "Censored"] = 0

    if censored is not None and censored != "" and censored != "all":
        filtered_df = filtered_df[filtered_df["Censored"] == int(censored)]

    available_types = (
        filtered_df["Type"].dropna().unique().tolist()
        if "Type" in filtered_df.columns
        else []
    )
    available_mdfs = (
        filtered_df["mdf"].dropna().unique().tolist()
        if "mdf" in filtered_df.columns
        else []
    )

    return {
        "df": filtered_df,
        "ui_data": {
            "rows": len(filtered_df),
            "equipment": equipment,
            "type": failure_type,
            "mdf": mdf,
            "censored_types": censored_types,
            "censored_mdfs": censored_mdfs,
            "available_types": available_types,
            "available_mdfs": available_mdfs,
        },
    }


def execute_weibull_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    df = inputs.get("df")
    if df is None:
        raise ValueError("El nodo Weibull requiere un conjunto de datos filtrado.")

    min_tbx = float(node.data.get("min_tbx", 0.0) or 0.0)
    min_ttx = float(node.data.get("min_ttx", 0.0) or 0.0)

    if min_tbx > 0 and "TBX" in df.columns:
        df = df[df["TBX"] >= min_tbx].copy()
    if min_ttx > 0 and "TTX" in df.columns:
        df = df[df["TTX"] >= min_ttx].copy()

    if df.empty or len(df) < 2:
        return {
            "error": "Insuficientes datos para ajustar Weibull (mínimo 2 registros)."
        }

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
            failures_df = df[df["Censored"] == 0] if "Censored" in df.columns else df
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
            "ks_p_value": round(res.get("p_value"), 4)
            if res.get("p_value") is not None
            else None,
            "applied_config": {"min_tbx": min_tbx, "min_ttx": min_ttx},
        }
    except Exception as e:
        return {"error": f"Fallo al ajustar Weibull: {str(e)}"}


def execute_kijima_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    df = inputs.get("df")
    if df is None:
        raise ValueError("El nodo Kijima requiere un conjunto de datos filtrado.")

    min_tbx = float(node.data.get("min_tbx", 0.0) or 0.0)
    min_ttx = float(node.data.get("min_ttx", 0.0) or 0.0)

    if min_tbx > 0 and "TBX" in df.columns:
        df = df[df["TBX"] >= min_tbx].copy()
    if min_ttx > 0 and "TTX" in df.columns:
        df = df[df["TTX"] >= min_ttx].copy()

    if df.empty or len(df) < 3:
        return {
            "error": "Insuficientes datos para ajustar Kijima (mínimo 3 registros)."
        }

    model_types_raw = node.data.get("model_types", None)
    if not model_types_raw:
        single_t = int(node.data.get("model_type", 1))
        model_types = [single_t]
    else:
        if isinstance(model_types_raw, list):
            model_types = [int(m) for m in model_types_raw if m is not None]
        else:
            model_types = [int(model_types_raw)]

    if not model_types:
        model_types = [1]

    try:
        column = "TBX" if "TBX" in df.columns else "Days"
        if column not in df.columns:
            column = "TTX" if "TTX" in df.columns else "Days"

        df_clean = df[df[column] > 0].copy()

        fitter = KijimaFitter()
        res_list = fitter.fit(
            dataframe=df_clean, column=column, censored_types=[], models=model_types
        )

        if not isinstance(res_list, list):
            res_list = [res_list] if res_list else []

        models_output = []
        for r in res_list:
            if not r:
                continue
            models_output.append(
                {
                    "model_name": r.get("model_name", "Kijima"),
                    "beta": round(r["beta"], 4) if r.get("beta") is not None else None,
                    "eta": round(r["eta"], 4) if r.get("eta") is not None else None,
                    "ar": round(r["ar"], 4) if r.get("ar") is not None else None,
                    "ap": round(r["ap"], 4) if r.get("ap") is not None else None,
                    "r2": round(r.get("ks_stat", 0.0), 4),
                }
            )

        first = models_output[0] if models_output else {}
        return {
            "models": models_output,
            "model_name": first.get("model_name", "Kijima"),
            "beta": first.get("beta"),
            "eta": first.get("eta"),
            "ar": first.get("ar"),
            "ap": first.get("ap"),
            "r2": first.get("r2"),
            "applied_config": {
                "model_types": model_types,
                "min_tbx": min_tbx,
                "min_ttx": min_ttx,
            },
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

        processed_records.append(
            {
                "component": r.get("component", ""),
                "mode": r.get("mode", ""),
                "effect": r.get("effect", ""),
                "severity": sev,
                "occurrence": occ,
                "detection": det,
                "rpn": rpn,
                "category": cat,
                "action": r.get("action", ""),
            }
        )

    return {"records": processed_records}


def execute_ram_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    df = inputs.get("df")
    if df is None:
        raise ValueError("El simulador RAM requiere un conjunto de datos.")

    if df.empty:
        raise ValueError("Dataset vacío en simulador RAM.")

    prev_eff = float(node.data.get("preventive_efficiency", 0.8))
    log_delay = float(node.data.get("logistics_delay", 4.0))

    # Uses the exact same ISO 20815 formula as /analysis/ram/simulate so that a
    # workbench-configured simulation and the standalone RAM tab always agree.
    return compute_ram_simulation(df, state.current_data, prev_eff, log_delay)


def execute_pareto_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    df = inputs.get("df")
    if df is None:
        raise ValueError("El análisis de Pareto requiere un conjunto de datos.")

    from src.reliability_analysis.analysis.pareto import ParetoAnalyzer

    try:
        group_by = node.data.get("group_by", "Equipment")
        if group_by.lower() in ("equipo", "equipment"):
            result = ParetoAnalyzer.analyze_by_equipment(df)
        elif group_by.lower() in ("tipo", "type"):
            result = ParetoAnalyzer.analyze_by_type(df)
        else:
            result = ParetoAnalyzer.analyze_by_failure_mode(df)

        vital, trivial, stats = ParetoAnalyzer.get_80_20_split(result)

        return {"group_by": group_by, "vital_few": vital[:5], "stats": stats}
    except Exception as e:
        return {"error": f"Fallo al calcular Pareto: {str(e)}"}


def execute_jackknife_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    df = inputs.get("df")
    if df is None:
        raise ValueError("El análisis Jackknife requiere un conjunto de datos.")

    try:
        group_col = node.data.get("compare_by", "Equipment")
        if group_col not in df.columns:
            group_col = "Equipment" if "Equipment" in df.columns else df.columns[0]

        if "TTX" not in df.columns:
            return {
                "error": "El dataset filtrado no contiene la columna 'TTX' (tiempo de detención) requerida para Jackknife."
            }

        stats = (
            df.groupby(group_col)
            .agg(
                failures=(group_col, "count"),
                total_downtime=("TTX", "sum"),
                avg_downtime=("TTX", "mean"),
            )
            .reset_index()
        )

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
            "acute_list": acute_items[:5],
            "compare_by": group_col,
        }
    except Exception as e:
        return {"error": f"Fallo al calcular Jackknife: {str(e)}"}


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
        "availability": round(availability, 2),
    }


def execute_criticality_node(
    node: NodeSchema, inputs: Dict[str, Any]
) -> Dict[str, Any]:
    df = inputs.get("df")
    if df is None:
        raise ValueError("El análisis de Criticidad requiere un conjunto de datos.")
    compare_by = node.data.get("compare_by", "mode")
    metric_x = node.data.get("metric_x", "count")
    try:
        result = compute_criticality(df, compare_by, metric_x)
    except ValueError as ve:
        return {"error": str(ve)}
    return result


def execute_event_plot_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    df = inputs.get("df")
    if df is None:
        raise ValueError("La Línea de Eventos requiere un conjunto de datos.")
    try:
        return compute_event_plot(df)
    except ValueError as ve:
        return {"error": str(ve)}


def execute_apm_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    df = inputs.get("df")
    if df is None:
        raise ValueError("El análisis APM requiere un conjunto de datos.")
    compare_by = node.data.get("compare_by", "equipment")
    try:
        bad_actors = compute_bad_actors(df, compare_by)
        return {"bad_actors": bad_actors, "compare_by": compare_by}
    except Exception as e:
        return {"error": f"Fallo al calcular APM: {str(e)}"}


def execute_rcm_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    df = inputs.get("df")
    equipment = resolve_node_equipment(node, df)
    if not equipment:
        return {
            "error": "Selecciona un equipo (o filtra por un único equipo aguas arriba) para generar las fichas RCM."
        }

    comments = []
    if df is not None and "Comment" in df.columns:
        eq_df = df[df["Equipment"] == equipment] if "Equipment" in df.columns else df
        comments = eq_df["Comment"].dropna().astype(str).tolist()

    rcm_sheets = LlmService.get_rcm_suggestions(equipment, comments)
    return {"standard": "SAE JA1011", "equipment": equipment, "rcm_sheets": rcm_sheets}


def execute_rca_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    df = inputs.get("df")
    equipment = resolve_node_equipment(node, df)
    if not equipment:
        return {
            "error": "Selecciona un equipo (o filtra por un único equipo aguas arriba) para generar el análisis RCA."
        }

    comments = []
    if df is not None and "Comment" in df.columns:
        eq_df = df[df["Equipment"] == equipment] if "Equipment" in df.columns else df
        comments = eq_df["Comment"].dropna().astype(str).tolist()

    rca_result = LlmService.get_rca_suggestions(equipment, comments)
    return {
        "standard": "IEC 62740",
        "equipment": equipment,
        "five_whys": rca_result.get("five_whys", []),
        "ishikawa": rca_result.get("ishikawa", {}),
    }


def execute_fta_node(node: NodeSchema, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes the top-event probability of a Fault Tree (IEC 61025) from a
    configurable gate (AND/OR) and a list of basic events, each with its own
    probability of occurrence. AND multiplies all probabilities (all events must
    occur simultaneously); OR is the probabilistic union 1 - product(1 - p_i).
    """
    gate_type = (node.data.get("gate_type") or "OR").upper()
    if gate_type not in ("AND", "OR"):
        gate_type = "OR"

    basic_events = node.data.get("basic_events") or []
    if not basic_events:
        basic_events = [
            {"name": "Falla de rodamiento", "probability": 0.12},
            {"name": "Falla de sello", "probability": 0.08},
            {"name": "Falla de motor", "probability": 0.05},
        ]

    probs = [float(e.get("probability", 0.0)) for e in basic_events]
    if gate_type == "AND":
        top_probability = 1.0
        for p in probs:
            top_probability *= p
    else:
        none_occur = 1.0
        for p in probs:
            none_occur *= 1.0 - p
        top_probability = 1.0 - none_occur

    return {
        "standard": "IEC 61025",
        "top_event": node.data.get("top_event", "Falla funcional del equipo"),
        "gate_type": gate_type,
        "basic_events": basic_events,
        "top_event_probability": round(top_probability, 6),
    }


# Core keyword-classification rules reused by the workbench's quick 'comment_mining'
# preview node (deep multi-model NLP comparison remains the job of the dedicated
# Copiloto IA tab, which calls /analysis/comment-mining directly).
_LEGACY_CATEGORY_KEYWORDS = [
    (
        "Operational",
        [
            "operacional",
            "operación",
            "decision",
            "decisión",
            "operational",
            "operation",
            "process",
            "operator",
        ],
    ),
    (
        "Cleaning/Blockage",
        [
            "limpieza",
            "atollo",
            "obstrucción",
            "obstruido",
            "cleaning",
            "blockage",
            "jam",
            "clog",
            "obstructed",
        ],
    ),
    (
        "Mechanical",
        [
            "mecánico",
            "mecanico",
            "perno",
            "shaft",
            "eje",
            "rodamiento",
            "bearing",
            "correa",
            "motor",
            "mechanical",
            "bolt",
            "belt",
        ],
    ),
    (
        "Electrical",
        [
            "eléctrico",
            "electrico",
            "cable",
            "bobina",
            "fase",
            "breaker",
            "contacto",
            "potencia",
            "electrical",
            "coil",
            "phase",
            "contact",
            "power",
        ],
    ),
    (
        "Instrumentation/Failure",
        [
            "falla",
            "alarma",
            "sensor",
            "calibracion",
            "calibración",
            "instrumentación",
            "instrumento",
            "failure",
            "alarm",
            "calibration",
            "instrumentation",
            "instrument",
        ],
    ),
]


def execute_comment_mining_node(
    node: NodeSchema, inputs: Dict[str, Any]
) -> Dict[str, Any]:
    df = inputs.get("df")
    if df is None or df.empty:
        return {
            "coverage": 0.0,
            "total_comments": 0,
            "categories": [],
            "model": "Legacy Keyword NLP",
        }

    comment_col = next(
        (c for c in df.columns if c.lower() in ("comentario", "comment")), None
    )
    if not comment_col:
        return {"error": "El dataset no contiene una columna de comentarios."}

    valid = df.dropna(subset=[comment_col])
    valid = valid[
        ~valid[comment_col]
        .astype(str)
        .str.lower()
        .isin(["---", "nan", "none", "null", "no aplica", "n/a"])
    ]

    total_records = len(df)
    total_comments = len(valid)
    coverage = (total_comments / total_records * 100.0) if total_records > 0 else 0.0

    type_col = "Type" if "Type" in df.columns else None
    records = []
    for _, row in valid.iterrows():
        text_lower = str(row[comment_col]).lower()
        category = "Others"
        for cat_name, keywords in _LEGACY_CATEGORY_KEYWORDS:
            if any(k in text_lower for k in keywords):
                category = cat_name
                break
        records.append(
            {
                "category": category,
                "type": str(row.get(type_col, "Unknown")) if type_col else "Unknown",
            }
        )

    categories = []
    for cat_name in [c for c, _ in _LEGACY_CATEGORY_KEYWORDS] + ["Others"]:
        cat_records = [r for r in records if r["category"] == cat_name]
        top_types = [
            t for t, _ in Counter([r["type"] for r in cat_records]).most_common(3)
        ]
        categories.append(
            {"category": cat_name, "count": len(cat_records), "top_types": top_types}
        )

    return {
        "model": node.data.get("model", "Legacy Keyword NLP"),
        "coverage": round(coverage, 2),
        "total_comments": total_comments,
        "categories": categories,
    }


def _result_from_output(output: Any, elapsed_ms: float) -> Dict[str, Any]:
    """
    Several execute_*_node functions signal a runtime failure (e.g. insufficient
    data, an unresolved equipment, a fitting exception) by returning a plain
    {"error": "..."} dict instead of raising, so the rest of the pipeline can
    keep running. This maps that convention onto the node's reported status,
    instead of unconditionally marking it "success" (which previously caused
    failed nodes to render as completed with the error tucked inside output).
    """
    if isinstance(output, dict) and "error" in output:
        return {
            "status": "error",
            "error": output["error"],
            "execution_ms": elapsed_ms,
        }
    return {"status": "success", "output": output, "execution_ms": elapsed_ms}


@router.post("/workbench/execute", tags=["Workbench"])
async def execute_pipeline(req: PipelineExecuteRequest) -> Dict[str, Any]:
    """Sorts and executes the analytical pipeline nodes sequentially with DAG multi-input support."""
    import time
    import pandas as pd

    try:
        execution_order = topological_sort(req.nodes, req.edges)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))

    # Build multi-parent connections graph mapping target -> list of sources
    parent_map: Dict[str, List[str]] = {}
    for e in req.edges:
        if e.target not in parent_map:
            parent_map[e.target] = []
        parent_map[e.target].append(e.source)

    results = {}
    context = {}  # Stores pandas DataFrames (df) passed between nodes

    for node in execution_order:
        start_time = time.time()
        node_id = node.id
        node_type = node.type

        # Resolve multi-parent inputs
        parent_ids = parent_map.get(node_id, [])
        node_inputs = {}

        dfs_to_combine = []
        for pid in parent_ids:
            if pid in context and "df" in context[pid]:
                dfs_to_combine.append(context[pid]["df"])

        if len(dfs_to_combine) == 1:
            node_inputs = {"df": dfs_to_combine[0]}
        elif len(dfs_to_combine) > 1:
            try:
                combined_df = pd.concat(
                    dfs_to_combine, ignore_index=True
                ).drop_duplicates()
                node_inputs = {"df": combined_df}
            except Exception as concat_err:
                logger.warning(
                    f"Could not concat DataFrames for node {node_id}: {concat_err}"
                )
                node_inputs = {"df": dfs_to_combine[0]}

        try:
            if node_type == "dataSource":
                res = execute_datasource_node(node, node_inputs)
                context[node_id] = {"df": res["df"]}
                elapsed = round((time.time() - start_time) * 1000, 2)
                results[node_id] = {
                    "status": "success",
                    "output": res["ui_data"],
                    "execution_ms": elapsed,
                }

            elif node_type == "filter":
                res = execute_filter_node(node, node_inputs)
                context[node_id] = {"df": res["df"]}
                elapsed = round((time.time() - start_time) * 1000, 2)
                results[node_id] = {
                    "status": "success",
                    "output": res["ui_data"],
                    "execution_ms": elapsed,
                }

            elif node_type == "weibull":
                res = execute_weibull_node(node, node_inputs)
                context[node_id] = {"df": node_inputs.get("df")}
                elapsed = round((time.time() - start_time) * 1000, 2)
                results[node_id] = _result_from_output(res, elapsed)

            elif node_type == "kijima":
                res = execute_kijima_node(node, node_inputs)
                context[node_id] = {"df": node_inputs.get("df")}
                elapsed = round((time.time() - start_time) * 1000, 2)
                results[node_id] = _result_from_output(res, elapsed)

            elif node_type == "fmeca":
                res = execute_fmeca_node(node, node_inputs)
                context[node_id] = {"df": node_inputs.get("df")}
                elapsed = round((time.time() - start_time) * 1000, 2)
                results[node_id] = _result_from_output(res, elapsed)

            elif node_type in ("ram", "ramSimulator"):
                res = execute_ram_node(node, node_inputs)
                context[node_id] = {"df": node_inputs.get("df")}
                elapsed = round((time.time() - start_time) * 1000, 2)
                results[node_id] = _result_from_output(res, elapsed)

            elif node_type == "pareto":
                res = execute_pareto_node(node, node_inputs)
                context[node_id] = {"df": node_inputs.get("df")}
                elapsed = round((time.time() - start_time) * 1000, 2)
                results[node_id] = _result_from_output(res, elapsed)

            elif node_type == "jackknife":
                res = execute_jackknife_node(node, node_inputs)
                context[node_id] = {"df": node_inputs.get("df")}
                elapsed = round((time.time() - start_time) * 1000, 2)
                results[node_id] = _result_from_output(res, elapsed)

            elif node_type == "trend":
                res = execute_trend_node(node, node_inputs)
                context[node_id] = {"df": node_inputs.get("df")}
                elapsed = round((time.time() - start_time) * 1000, 2)
                results[node_id] = _result_from_output(res, elapsed)

            elif node_type == "criticality":
                res = execute_criticality_node(node, node_inputs)
                context[node_id] = {"df": node_inputs.get("df")}
                elapsed = round((time.time() - start_time) * 1000, 2)
                results[node_id] = _result_from_output(res, elapsed)

            elif node_type == "event_plot":
                res = execute_event_plot_node(node, node_inputs)
                context[node_id] = {"df": node_inputs.get("df")}
                elapsed = round((time.time() - start_time) * 1000, 2)
                results[node_id] = _result_from_output(res, elapsed)

            elif node_type == "apm":
                res = execute_apm_node(node, node_inputs)
                context[node_id] = {"df": node_inputs.get("df")}
                elapsed = round((time.time() - start_time) * 1000, 2)
                results[node_id] = _result_from_output(res, elapsed)

            elif node_type == "rcm":
                res = execute_rcm_node(node, node_inputs)
                context[node_id] = {"df": node_inputs.get("df")}
                elapsed = round((time.time() - start_time) * 1000, 2)
                results[node_id] = _result_from_output(res, elapsed)

            elif node_type == "rca":
                res = execute_rca_node(node, node_inputs)
                context[node_id] = {"df": node_inputs.get("df")}
                elapsed = round((time.time() - start_time) * 1000, 2)
                results[node_id] = _result_from_output(res, elapsed)

            elif node_type == "fta":
                res = execute_fta_node(node, node_inputs)
                context[node_id] = {"df": node_inputs.get("df")}
                elapsed = round((time.time() - start_time) * 1000, 2)
                results[node_id] = _result_from_output(res, elapsed)

            elif node_type == "comment_mining":
                res = execute_comment_mining_node(node, node_inputs)
                context[node_id] = {"df": node_inputs.get("df")}
                elapsed = round((time.time() - start_time) * 1000, 2)
                results[node_id] = _result_from_output(res, elapsed)

            else:
                results[node_id] = {
                    "status": "error",
                    "error": f"Tipo de nodo '{node_type}' no soportado.",
                }

        except Exception as node_err:
            logger.error(
                f"Error executing node {node_id} ({node_type}): {str(node_err)}"
            )
            results[node_id] = {"status": "error", "error": str(node_err)}

    # Generate structured execution logs for the UI diagnostic console
    import datetime

    current_logs = []
    for nid, r in results.items():
        n_type = next((n.type for n in req.nodes if n.id == nid), "desconocido")
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        if r.get("status") == "success":
            out = r.get("output", {})
            msg = f"Ejecución exitosa en {r.get('execution_ms', 0)}ms."
            if isinstance(out, dict):
                if "rows" in out:
                    msg += f" {out['rows']} filas procesadas."
                elif "beta" in out and "eta" in out:
                    msg += f" Parámetros Weibull: β={out['beta']}, η={out['eta']}."
                elif "models" in out:
                    msg += f" {len(out['models'])} modelos Kijima ajustados."
                elif "availability" in out:
                    msg += f" Disponibilidad RAM: {out['availability']}%."
                elif "top_event_probability" in out:
                    msg += f" Probabilidad evento tope FTA: {out['top_event_probability'] * 100:.2f}%."
                elif "rcm_sheets" in out:
                    msg += f" {len(out['rcm_sheets'])} fichas RCM generadas para {out.get('equipment', '')}."
                elif "five_whys" in out:
                    msg += f" Análisis RCA generado para {out.get('equipment', '')}."
                elif "bad_actors" in out:
                    msg += f" {len(out['bad_actors'])} equipos rankeados."
                elif "categories" in out:
                    msg += f" Cobertura de comentarios: {out.get('coverage', 0)}%."
            current_logs.append(
                {
                    "id": f"log-{nid}-{datetime.datetime.now().timestamp()}",
                    "timestamp": ts,
                    "node_id": nid,
                    "node_type": n_type,
                    "level": "INFO",
                    "message": msg,
                }
            )
        else:
            current_logs.append(
                {
                    "id": f"log-{nid}-{datetime.datetime.now().timestamp()}",
                    "timestamp": ts,
                    "node_id": nid,
                    "node_type": n_type,
                    "level": "ERROR",
                    "message": f"Fallo de ejecución: {r.get('error', 'Error no especificado')}",
                }
            )

    state.workbench_logs.extend(current_logs)
    # Keep last 100 log entries
    state.workbench_logs = state.workbench_logs[-100:]

    return {"status": "success", "results": results, "logs": current_logs}


@router.get("/workbench/logs", tags=["Workbench"])
async def get_workbench_logs() -> Dict[str, Any]:
    """Returns stored execution logs for the current session."""
    return {"status": "success", "logs": state.workbench_logs}


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
            "edges": [e.model_dump() for e in req.edges],
        }

        with open(PIPELINES_FILE, "w", encoding="utf-8") as f:
            json.dump(pipelines, f, indent=2, ensure_ascii=False)

        return {
            "status": "success",
            "message": f"Pipeline '{req.name}' guardado correctamente.",
        }
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
            raise HTTPException(
                status_code=404, detail=f"Pipeline '{name}' no encontrado."
            )

        return {"status": "success", "pipeline": pipelines[name]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading pipeline: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
