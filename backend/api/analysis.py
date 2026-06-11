from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import pandas as pd
import numpy as np
import logging
import traceback

import state
from models.requests import (
    ParetoRequest,
    AnalysisRequest,
    WeibullFitRequest,
    OptimalPMRequest,
    ConditionalReliabilityRequest,
)
from src.reliability_analysis.analysis.pareto import ParetoAnalyzer
from src.reliability_analysis.analysis.models import ReliabilityFitter

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/analysis/pareto", tags=["Analysis"])
async def pareto_analysis(req: ParetoRequest) -> Dict[str, Any]:
    if state.filter_manager is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        data = state.filter_manager.get_filtered_data()

        eq = req.equipment if req.equipment else None
        ft = req.failure_type if req.failure_type else None

        if data.empty:
            return {
                "status": "warning",
                "message": "No data matches current filters",
                "pareto": None,
            }

        if req.group_by.lower() in ("equipo", "equipment"):
            result = ParetoAnalyzer.analyze_by_equipment(data)
        elif req.group_by.lower() in ("tipo", "type"):
            result = ParetoAnalyzer.analyze_by_type(data, equipment=eq)
        elif req.group_by.lower() == "mdf":
            result = ParetoAnalyzer.analyze_by_failure_mode(
                data, equipment=eq, failure_type=ft
            )
        else:
            raise ValueError(f"Unknown group_by: {req.group_by}")

        vital, trivial, stats = ParetoAnalyzer.get_80_20_split(result)

        return {
            "status": "success",
            "group_by": req.group_by,
            "pareto": result,
            "analysis": {"vital_few": vital, "trivial_many": trivial, "stats": stats},
        }
    except Exception as e:
        logger.error(f"Pareto analysis error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analysis/jackknife", tags=["Analysis"])
async def jackknife_analysis(req: AnalysisRequest) -> Dict[str, Any]:
    if state.current_data is None or state.filter_manager is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        data = state.filter_manager.get_filtered_data()

        if req.types_to_use:
            data = data[data["Type"].isin(req.types_to_use)]

        if data.empty:
            raise HTTPException(status_code=400, detail="Filtered data is empty")

        if req.compare_by == "equipment":
            group_col = "Equipment"
        elif req.compare_by == "type":
            group_col = "Type"
        elif req.compare_by == "mode":
            group_col = "mdf"
        else:
            group_col = "Equipment"

        if group_col not in data.columns:
            group_col = "Equipment"

        stats = (
            data.groupby(group_col)
            .agg(
                failures=(group_col, "count"),
                total_downtime=("TTX", "sum"),
                avg_downtime=("TTX", "mean"),
            )
            .reset_index()
        )

        total_failures = float(stats["failures"].sum())

        avg_failures = float(stats["failures"].mean()) if not stats.empty else 0
        avg_prob = (avg_failures / total_failures) if total_failures > 0 else 0
        avg_total = float(stats["total_downtime"].mean()) if not stats.empty else 0
        avg_mean = float(stats["avg_downtime"].mean()) if not stats.empty else 0

        scatter_data = [
            {
                "name": str(row[group_col]),
                "x": float(row["failures"]),
                "x_prob": float(row["failures"]) / total_failures
                if total_failures > 0
                else 0.0,
                "y_total": float(row["total_downtime"]),
                "y_avg": float(row["avg_downtime"]),
            }
            for _, row in stats.iterrows()
        ]

        return {
            "status": "success",
            "scatter_data": scatter_data,
            "averages": {
                "failures": avg_failures,
                "probability": avg_prob,
                "total_downtime": avg_total,
                "avg_downtime": avg_mean,
            },
        }
    except Exception as e:
        logger.error(f"Jackknife analysis error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analysis/fit", tags=["Analysis"])
async def fit_data(req: WeibullFitRequest) -> Dict[str, Any]:
    if state.current_data is None or state.filter_manager is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        data = state.filter_manager.get_filtered_data()

        if req.types_to_fit:
            data = data[data["Type"].isin(req.types_to_fit)]

        if data.empty:
            return {
                "status": "error",
                "message": "Filtered data is empty",
                "parameters": None,
            }

        fit_data = data.copy()
        target_col = req.target_column or "TBX"

        if target_col == "TTX" and "TTX" not in fit_data.columns:
            fit_data["TTX"] = 0.0
        elif (
            target_col == "TBX"
            and "Days" in fit_data.columns
            and "TBX" not in fit_data.columns
        ):
            target_col = "Days"

        if target_col in fit_data.columns and target_col != "TBX":
            fit_data["TBX"] = fit_data[target_col]

        fitter = ReliabilityFitter(fit_data)
        results = fitter.fit_weibull(
            column=target_col, censored_failure_types=req.censored_failure_types
        )

        if not results or results.get("beta") is None:
            return {
                "status": "error",
                "message": "Could not fit Weibull distribution",
                "parameters": None,
            }

        curve_data = None
        try:
            from reliability.Distributions import Weibull_Distribution

            if results.get("beta") and results.get("eta"):
                dist = Weibull_Distribution(alpha=results["eta"], beta=results["beta"])
                t_max = (
                    float(fit_data["TBX"].max())
                    if "TBX" in fit_data.columns
                    else 1000.0
                )
                times = np.linspace(1e-3, t_max * 1.2, 100)

                curve_data = {"time": times.tolist()}
                if req.target_column == "TTX":
                    curve_data["pdf"] = dist.PDF(xvals=times).tolist()
                    curve_data["cdf"] = dist.CDF(xvals=times).tolist()
                else:
                    curve_data["reliability"] = dist.SF(xvals=times).tolist()
                    curve_data["hazard_rate"] = dist.HF(xvals=times).tolist()
        except Exception as ex:
            logger.warning(f"Error generating reliability curve: {ex}")

        return {
            "status": "success",
            "parameters": {
                "beta": float(results["beta"]),
                "eta": float(results["eta"]),
            },
            "goodness_of_fit": {
                "aic": float(results.get("aic")),
                "bic": float(results.get("bic")),
            },
            "reliability_curve": curve_data,
            "sample_size": len(data),
            "failures_count": results.get("failures_count"),
            "censored_count": results.get("censored_count"),
        }
    except Exception as e:
        logger.error(f"Fit analysis error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analysis/bad-actors", tags=["Analysis"])
async def bad_actors_analysis(req: AnalysisRequest) -> Dict[str, Any]:
    if state.filter_manager is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        data = state.filter_manager.get_filtered_data()
        if data.empty:
            return {"status": "warning", "bad_actors": []}

        group_col = "Equipment" if req.compare_by == "equipment" else "Type"

        uptime_col = (
            "TBX"
            if "TBX" in data.columns
            else ("Days" if "Days" in data.columns else None)
        )
        downtime_col = "TTX" if "TTX" in data.columns else None

        stats = []
        for name, group in data.groupby(group_col):
            failures = len(group)
            downtime = float(group[downtime_col].sum()) if downtime_col else 0.0
            uptime = float(group[uptime_col].sum()) if uptime_col else 0.0

            mttr = downtime / failures if failures > 0 else 0.0
            mtbf = uptime / failures if failures > 0 else 0.0
            availability = (
                (uptime / (uptime + downtime)) * 100 if (uptime + downtime) > 0 else 0.0
            )

            stats.append(
                {
                    "name": str(name),
                    "failures": failures,
                    "downtime": downtime,
                    "mttr": mttr,
                    "mtbf": mtbf,
                    "availability": availability,
                }
            )

        stats.sort(key=lambda x: x["downtime"], reverse=True)
        return {"status": "success", "bad_actors": stats[:50]}
    except Exception as e:
        logger.error(f"Bad actors error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analysis/growth", tags=["Analysis"])
async def reliability_growth(req: AnalysisRequest) -> Dict[str, Any]:
    if state.filter_manager is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        data = state.filter_manager.get_filtered_data()
        if data.empty:
            return {
                "status": "warning",
                "cumulative_time": [],
                "cumulative_failures": [],
            }

        if "Start_Date" in data.columns:
            data = data.sort_values("Start_Date")

        if "TBX" in data.columns:
            uptime_col = "TBX"
        elif "Days" in data.columns:
            uptime_col = "Days"
        else:
            uptime_col = "TTX" if "TTX" in data.columns else None

        if uptime_col and uptime_col in data.columns:
            cum_time = data[uptime_col].cumsum().tolist()
            cum_failures = list(range(1, len(data) + 1))
            return {
                "status": "success",
                "cumulative_time": cum_time,
                "cumulative_failures": cum_failures,
            }
        return {"status": "error", "detail": "Time column not found"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analysis/event-plot", tags=["Analysis"])
async def event_plot(req: AnalysisRequest) -> Dict[str, Any]:
    if state.filter_manager is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        data = state.filter_manager.get_filtered_data()
        if "Start_Date" not in data.columns:
            raise HTTPException(
                status_code=400, detail="Dataset must contain 'Start_Date' column."
            )

        if data.empty:
            return {
                "status": "success",
                "events": {},
                "min_date": None,
                "max_date": None,
            }

        data["Start_Date"] = pd.to_datetime(data["Start_Date"])
        min_val = data["Start_Date"].min()
        min_date = min_val.strftime("%Y-%m-%d") if pd.notnull(min_val) else None

        if "End_Date" in data.columns:
            data["End_Date"] = pd.to_datetime(data["End_Date"])
            max_val = data["End_Date"].max()
            max_date = max_val.strftime("%Y-%m-%d") if pd.notnull(max_val) else None
        else:
            data["End_Date"] = data["Start_Date"]
            max_val = data["Start_Date"].max()
            max_date = max_val.strftime("%Y-%m-%d") if pd.notnull(max_val) else None

        event_data = {}
        for name, group in data.groupby("Equipment"):
            event_list = []
            for _, row in group.iterrows():
                if pd.notnull(row["Start_Date"]):
                    start_str = row["Start_Date"].strftime("%Y-%m-%dT%H:%M:%S")
                    end_val = (
                        row["End_Date"]
                        if pd.notnull(row["End_Date"])
                        else row["Start_Date"]
                    )
                    end_str = end_val.strftime("%Y-%m-%dT%H:%M:%S")
                    mode_val = str(row["mdf"]) if "mdf" in row else "Unknown"
                    type_val = str(row["Type"]) if "Type" in row else "Unknown"
                    event_list.append(
                        {
                            "start": start_str,
                            "end": end_str,
                            "mode": mode_val,
                            "type": type_val,
                        }
                    )
            event_data[str(name)] = event_list

        return {
            "status": "success",
            "events": event_data,
            "min_date": min_date,
            "max_date": max_date,
        }
    except Exception as e:
        logger.error(f"Event plot error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analysis/optimal-pm", tags=["Analysis"])
async def optimal_pm(req: OptimalPMRequest) -> Dict[str, Any]:
    try:
        fit_response = await fit_data(req)
        params = fit_response.get("parameters")
        if not params or not params.get("beta"):
            raise HTTPException(
                status_code=400, detail="Could not fit Weibull to calculate Optimal PM."
            )

        if params["beta"] <= 1.0:
            return {"status": "success", "optimal_pm_interval": None}

        from reliability.PoF import optimal_replacement_time
        from reliability.Distributions import Weibull_Distribution

        dist = Weibull_Distribution(alpha=params["eta"], beta=params["beta"])

        try:
            optimal_time = optimal_replacement_time(
                cost_PM=req.cost_pm,
                cost_failure=req.cost_failure,
                weibull_distribution=dist,
                show_plot=False,
            )
        except ValueError:
            optimal_time = None

        import math

        if (
            optimal_time is None
            or math.isnan(float(optimal_time))
            or math.isinf(float(optimal_time))
        ):
            opt_val = None
        else:
            opt_val = float(optimal_time)

        return {"status": "success", "optimal_pm_interval": opt_val}
    except Exception as e:
        logger.error(f"Optimal PM error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analysis/conditional-reliability", tags=["Analysis"])
async def conditional_reliability(req: ConditionalReliabilityRequest) -> Dict[str, Any]:
    try:
        fit_response = await fit_data(req)
        params = fit_response.get("parameters")
        if not params or not params.get("beta"):
            raise HTTPException(
                status_code=400,
                detail="Could not fit Weibull to calculate conditional reliability.",
            )

        from reliability.Distributions import Weibull_Distribution

        dist = Weibull_Distribution(alpha=params["eta"], beta=params["beta"])

        # Confiabilidad condicional absoluta: R(T_target) / R(T_current)
        if req.mission_time < req.current_age:
            prob_success = 1.0  # Ya sobrevivió a este tiempo
        else:
            R_T = float(dist.SF(req.current_age))
            if R_T <= 1e-10:
                prob_success = 0.0
            else:
                prob_success = float(dist.SF(req.mission_time)) / R_T

        import math

        if math.isnan(prob_success) or prob_success < 0:
            prob_success = 0.0
        elif prob_success > 1.0:
            prob_success = 1.0

        return {
            "status": "success",
            "success_probability": float(prob_success),
            "failure_probability": 1.0 - float(prob_success),
        }
    except Exception as e:
        logger.error(f"Conditional reliability error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analysis/kpi-trend", tags=["Analysis"])
async def kpi_trend(req: AnalysisRequest) -> Dict[str, Any]:
    if state.filter_manager is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        data = state.filter_manager.get_filtered_data()
        if data.empty:
            return {"status": "success", "trend": []}

        if "Start_Date" not in data.columns:
            raise HTTPException(
                status_code=400,
                detail="Dataset must contain datetime start column (Start_Date).",
            )

        df = data.copy()
        df = df.dropna(subset=["Start_Date"])
        if df.empty:
            return {"status": "success", "trend": []}

        df["month_period"] = df["Start_Date"].dt.to_period("M")

        grouped = df.groupby("month_period")
        trend_data = []

        for period in sorted(grouped.groups.keys()):
            group = grouped.get_group(period)

            failures = int(len(group))
            downtime = float(group["TTX"].sum()) if "TTX" in group.columns else 0.0
            uptime = float(group["TBX"].sum()) if "TBX" in group.columns else 0.0

            mtbf = float(uptime / failures) if failures > 0 else 0.0
            mttr = float(downtime / failures) if failures > 0 else 0.0

            total_time = uptime + downtime
            availability = (
                float((uptime / total_time) * 100.0) if total_time > 0.0 else 0.0
            )

            trend_data.append(
                {
                    "month": str(period),
                    "failures": failures,
                    "downtime": downtime,
                    "uptime": uptime,
                    "mtbf": mtbf,
                    "mttr": mttr,
                    "availability": availability,
                }
            )

        return {"status": "success", "trend": trend_data}
    except Exception as e:
        logger.error(f"KPI trend analysis error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analysis/comment-mining", tags=["Analysis"])
async def comment_mining(req: AnalysisRequest) -> Dict[str, Any]:
    if state.filter_manager is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        data = state.filter_manager.get_filtered_data()
        if data.empty:
            return {
                "status": "success",
                "keywords": [],
                "categories": [],
                "coverage": 0.0,
            }

        comment_col = None
        for col in data.columns:
            if col.lower() in ("comentario", "comment"):
                comment_col = col
                break

        if not comment_col:
            return {
                "status": "warning",
                "message": "No comment column found in dataset",
                "keywords": [],
                "categories": [],
            }

        comments = data[comment_col].dropna().astype(str)
        comments = comments[
            ~comments.str.lower().isin(
                ["---", "nan", "none", "null", "no aplica", "n/a"]
            )
        ]

        total_records = len(data)
        comments_count = len(comments)
        coverage = (
            (comments_count / total_records * 100.0) if total_records > 0 else 0.0
        )

        if comments.empty:
            return {
                "status": "success",
                "keywords": [],
                "categories": [],
                "coverage": 0.0,
                "total_comments": 0,
            }

        # Core stop words in both English and Spanish
        core_stop_words = {
            "de",
            "la",
            "el",
            "y",
            "en",
            "por",
            "que",
            "para",
            "un",
            "una",
            "con",
            "se",
            "no",
            "del",
            "al",
            "lo",
            "los",
            "las",
            "es",
            "su",
            "sus",
            "como",
            "más",
            "mas",
            "o",
            "pero",
            "este",
            "esta",
            "ha",
            "debido",
            "realiza",
            "presenta",
            "genera",
            "causa",
            "además",
            "ademas",
            "sobre",
            "entre",
            "otro",
            "otra",
            "otros",
            "otras",
            "caso",
            "hace",
            "hecho",
            "estos",
            "estas",
            "sino",
            "toda",
            "todo",
            "todos",
            "todas",
            "donde",
            "desde",
            "hasta",
            "cuando",
            "quien",
            "cual",
            "cuales",
            "muy",
            "sólo",
            "solo",
            "cada",
            "bien",
            "también",
            "tambien",
            "tampoco",
            "después",
            "despues",
            "antes",
            "ahora",
            "the",
            "of",
            "and",
            "to",
            "in",
            "is",
            "you",
            "that",
            "it",
            "he",
            "was",
            "for",
            "on",
            "are",
            "as",
            "with",
            "his",
            "they",
            "i",
            "at",
            "be",
            "this",
            "have",
            "from",
            "or",
            "one",
            "had",
            "by",
            "but",
            "not",
            "what",
            "all",
            "were",
            "we",
            "when",
            "your",
            "can",
            "said",
            "there",
            "use",
            "an",
            "each",
            "which",
            "she",
            "do",
            "how",
            "their",
            "if",
            "will",
            "up",
            "other",
            "about",
            "out",
            "many",
            "then",
        }

        # Generic terms we want to exclude from standalone keywords or joint bigrams (uninformative)
        generic_words = {
            "decision",
            "decisión",
            "operacional",
            "operacion",
            "operación",
            "detencion",
            "detención",
            "detenida",
            "detenido",
            "detencin",
            "proceso",
            "procesos",
            "sistema",
            "sistemas",
            "falla",
            "fallas",
            "parada",
            "paradas",
            "bajo",
            "alto",
            "nivel",
            "niveles",
            "vacio",
            "vacío",
            "corriendo",
            "tiempo",
            "tiempos",
            "minuto",
            "minutos",
            "hora",
            "horas",
            "días",
            "día",
            "sección",
            "área",
            "operario",
            "operarios",
            "turno",
            "turnos",
            "reporte",
            "reportes",
            "observación",
            "observaciones",
            "trabajo",
            "trabajos",
            "personal",
            "inspección",
            "inspeccion",
            "revisión",
            "revision",
            "registro",
            "registros",
            "código",
            "codigo",
            "estado",
            "estados",
            "actividad",
            "actividades",
            "inicio",
            "fin",
            "espera",
            "esperas",
            "valor",
            "valores",
            "equipo",
            "equipos",
            "problema",
            "problemas",
            "evento",
            "eventos",
            "motivo",
            "motivos",
            "failure",
            "failures",
            "stop",
            "stops",
            "stopped",
            "system",
            "systems",
            "downtime",
            "uptime",
            "running",
            "code",
            "codes",
            "activity",
            "activities",
            "waiting",
            "process",
            "operational",
            "operation",
            "operations",
            "area",
            "shift",
            "shifts",
            "operator",
            "report",
            "reports",
            "observation",
            "observations",
            "inspection",
            "inspections",
        }

        def clean_word(word: str) -> str:
            word = word.lower()
            glitches = {
                "detencin": "detencion",
                "proteccin": "proteccion",
                "calibracin": "calibracion",
                "operacin": "operacion",
                "decisin": "decision",
                "observacin": "observacion",
                "inspeccin": "inspeccion",
                "revisin": "revision",
                "obstruccin": "obstruccion",
                "ventilacin": "ventilacion",
                "alimentacin": "alimentacion",
            }
            return glitches.get(word, word)

        def reduce_plural(word: str) -> str:
            if word.endswith("ces"):
                return word[:-3] + "z"
            elif word.endswith("es") and len(word) > 4:
                candidate = word[:-2]
                if candidate[-1] in "bcdfghjklmnpqrstvwxyz":
                    return candidate
            elif word.endswith("s") and len(word) > 3:
                if not word.endswith("is") and not word.endswith("us"):
                    return word[:-1]
            return word

        from collections import Counter
        import re

        all_terms = []
        analyzed_records = []

        type_col = "Type" if "Type" in data.columns else None
        mode_col = "mdf"

        for _, row in data.dropna(subset=[comment_col]).iterrows():
            text = str(row[comment_col])
            text_lower = text.lower()
            if text_lower in ["---", "nan", "none", "null", "no aplica", "n/a"]:
                continue

            cat = "Others"
            if any(
                k in text_lower
                for k in [
                    "operacional",
                    "operación",
                    "decision",
                    "decisión",
                    "operational",
                    "operation",
                    "process",
                    "operator",
                ]
            ):
                cat = "Operational"
            elif any(
                k in text_lower
                for k in [
                    "limpieza",
                    "atollo",
                    "obstrucción",
                    "obstruido",
                    "cleaning",
                    "blockage",
                    "jam",
                    "clog",
                    "obstructed",
                ]
            ):
                cat = "Cleaning/Blockage"
            elif any(
                k in text_lower
                for k in [
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
                ]
            ):
                cat = "Mechanical"
            elif any(
                k in text_lower
                for k in [
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
                ]
            ):
                cat = "Electrical"
            elif any(
                k in text_lower
                for k in [
                    "falla",
                    "alarma",
                    "sensor",
                    "calibracion",
                    "calibración",
                    "instrumentación",
                    "instrumento",
                    "failure",
                    "alarm",
                    "sensor",
                    "calibration",
                    "instrumentation",
                    "instrument",
                ]
            ):
                cat = "Instrumentation/Failure"

            t_val = str(row[type_col]) if type_col else "Unknown"
            m_val = str(row[mode_col]) if mode_col in row else "Unknown"

            analyzed_records.append({"category": cat, "type": t_val, "mode": m_val})

            # Words extraction
            words = re.findall(r"\b[a-zA-Záéíóúñ]{3,}\b", text_lower)
            cleaned_words = []

            for w in words:
                cleaned = clean_word(w)
                norm = reduce_plural(cleaned)
                cleaned_words.append(norm)

            # Extract unigrams
            for w in cleaned_words:
                if w not in core_stop_words and w not in generic_words and len(w) > 2:
                    all_terms.append(w)

            # Extract bigrams
            for i in range(len(cleaned_words) - 1):
                w1, w2 = cleaned_words[i], cleaned_words[i + 1]
                if w1 not in core_stop_words and w2 not in core_stop_words:
                    if w1 not in generic_words or w2 not in generic_words:
                        bigram = f"{w1} {w2}"
                        all_terms.append(bigram)

        counter = Counter(all_terms)
        top_keywords = [
            {"word": word, "count": count} for word, count in counter.most_common(15)
        ]

        # Aggregate category details with cross-tabulation metadata
        categories_details = []

        for cat_name in [
            "Operational",
            "Cleaning/Blockage",
            "Mechanical",
            "Electrical",
            "Instrumentation/Failure",
            "Others",
        ]:
            cat_records = [r for r in analyzed_records if r["category"] == cat_name]
            count = len(cat_records)

            top_types = [
                t for t, _ in Counter([r["type"] for r in cat_records]).most_common(3)
            ]
            top_modes = [
                m for m, _ in Counter([r["mode"] for r in cat_records]).most_common(3)
            ]

            categories_details.append(
                {
                    "category": cat_name,
                    "count": count,
                    "top_types": top_types,
                    "top_modes": top_modes,
                }
            )

        return {
            "status": "success",
            "coverage": float(coverage),
            "total_comments": int(comments_count),
            "keywords": top_keywords,
            "categories": categories_details,
        }
    except Exception as e:
        logger.error(f"Comment mining error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
