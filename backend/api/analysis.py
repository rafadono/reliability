from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import pandas as pd
import numpy as np
import logging
import traceback

import time
from src.reliability_analysis.utils.config import NLP_MODELS_TO_COMPARE
from src.reliability_analysis.analysis.hf_classifier import SemanticModelManager
from huggingface_hub import scan_cache_dir

import state
from models.requests import (
    ParetoRequest,
    AnalysisRequest,
    WeibullFitRequest,
    OptimalPMRequest,
    ConditionalReliabilityRequest,
    CriticalityRequest,
    KijimaFitRequest,
    KpiTrendRequest,
    RcmSuggestRequest,
    FmecaRpnRequest,
    RamSimulateRequest,
    RcaAnalysisRequest,
)
from src.reliability_analysis.analysis.pareto import ParetoAnalyzer
from src.reliability_analysis.analysis.models import ReliabilityFitter, KijimaFitter
from services.llm import LlmService

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


@router.post("/analysis/jackknife-plot", tags=["Analysis"])
async def jackknife_plot_analysis(req: AnalysisRequest) -> Dict[str, Any]:
    """
    Get scatter plot data (frequency vs total downtime) and classified regions for the Maintenance Jackknife diagram.
    Note: This does not perform statistical Leave-One-Out resampling (Jackknife confidence intervals),
    but groups and aggregates failure records to identify chronic and acute bad actors.
    """
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

        scatter_data = []
        for _, row in stats.iterrows():
            item_failures = float(row["failures"])
            item_downtime = float(row["total_downtime"])
            item_avg_downtime = float(row["avg_downtime"])
            item_prob = item_failures / total_failures if total_failures > 0 else 0.0

            scatter_data.append({
                "name": str(row[group_col]),
                "x": item_failures,
                "x_prob": item_prob,
                "y_total": item_downtime,
                "y_avg": item_avg_downtime,
            })

        # Calculate regions on backend (all business/analytical calculations here)
        regions = {
            "acuteChronic": [],
            "acute": [],
            "chronic": [],
            "acceptable": []
        }

        for item in scatter_data:
            x = item["x"]
            y = item["y_total"]
            if x > avg_failures and y > avg_total:
                regions["acuteChronic"].append(item)
            elif x <= avg_failures and y > avg_total:
                regions["acute"].append(item)
            elif x > avg_failures and y <= avg_total:
                regions["chronic"].append(item)
            else:
                regions["acceptable"].append(item)

        return {
            "status": "success",
            "scatter_data": scatter_data,
            "averages": {
                "failures": avg_failures,
                "probability": avg_prob,
                "total_downtime": avg_total,
                "avg_downtime": avg_mean,
            },
            "regions": regions,
        }
    except Exception as e:
        logger.error(f"Jackknife plot analysis error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analysis/criticality-plot", tags=["Analysis"])
async def criticality_plot_analysis(req: CriticalityRequest) -> Dict[str, Any]:
    """
    Get scatter plot data (frequency/probability vs average downtime) and classified regions for the Criticality Matrix.
    """
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

        scatter_data = []
        for _, row in stats.iterrows():
            item_failures = float(row["failures"])
            item_downtime = float(row["total_downtime"])
            item_avg_downtime = float(row["avg_downtime"])
            item_prob = item_failures / total_failures if total_failures > 0 else 0.0

            scatter_data.append({
                "name": str(row[group_col]),
                "x": item_failures,
                "x_prob": item_prob,
                "y_total": item_downtime,
                "y_avg": item_avg_downtime,
            })

        # Calculate regions on backend (all business/analytical calculations here)
        metric_x = req.metric_x or "count"
        if metric_x == "probability":
            avg_x = avg_prob * 100.0
        else:
            avg_x = avg_failures
        avg_y = avg_mean

        regions = {
            "highRisk": [],
            "highConsequence": [],
            "highFrequency": [],
            "lowRisk": []
        }

        for item in scatter_data:
            x_val = item["x_prob"] * 100.0 if metric_x == "probability" else item["x"]
            y_val = item["y_avg"]

            item_with_val = {**item, "x_val": x_val}

            if x_val > avg_x and y_val > avg_y:
                regions["highRisk"].append(item_with_val)
            elif x_val <= avg_x and y_val > avg_y:
                regions["highConsequence"].append(item_with_val)
            elif x_val > avg_x and y_val <= avg_y:
                regions["highFrequency"].append(item_with_val)
            else:
                regions["lowRisk"].append(item_with_val)

        return {
            "status": "success",
            "scatter_data": scatter_data,
            "averages": {
                "failures": avg_failures,
                "probability": avg_prob,
                "total_downtime": avg_total,
                "avg_downtime": avg_mean,
            },
            "regions": regions,
        }
    except Exception as e:
        logger.error(f"Criticality plot analysis error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))



@router.post("/analysis/fit", tags=["Analysis"])
async def fit_data(req: WeibullFitRequest) -> Dict[str, Any]:
    if state.current_data is None or state.filter_manager is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        data = state.filter_manager.get_filtered_data()

        if req.types_to_fit:
            all_types = list(req.types_to_fit)
            if req.censored_failure_types:
                for t in req.censored_failure_types:
                    if t not in all_types:
                        all_types.append(t)
            data = data[data["Type"].isin(all_types)]

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

        min_tbx = float(req.min_tbx) if req.min_tbx is not None else 0.0
        excluded_idxs = req.excluded_indices or []

        # Build the intervals list for output
        intervals_df = fit_data[fit_data["TBX"] >= 0.0].copy()
        if "Start_Date" in intervals_df.columns:
            intervals_df = intervals_df.sort_values("Start_Date")
        
        intervals_list = []
        active_idx = 1
        indices_to_drop = []
        for i, (orig_idx, row) in enumerate(intervals_df.iterrows()):
            tbx_val = float(row["TBX"])
            is_baseline_start = (i == 0)
            
            is_manually_excluded = False
            if not is_baseline_start:
                is_manually_excluded = (active_idx in excluded_idxs)
                
            is_included = (tbx_val >= min_tbx) and (tbx_val > 0.0) and not is_baseline_start and not is_manually_excluded
            if is_manually_excluded:
                indices_to_drop.append(orig_idx)
            
            date_str = str(row["Start_Date"]) if "Start_Date" in row and pd.notnull(row["Start_Date"]) else "-"
            type_val = str(row["Type"]) if "Type" in row else "-"
            mode_val = str(row["mdf"]) if "mdf" in row else "-"
            
            intervals_list.append({
                "index": active_idx,
                "date": date_str,
                "tbx": tbx_val,
                "type": type_val,
                "mode": mode_val,
                "included": is_included,
                "is_baseline": is_baseline_start,
                "manually_excluded": is_manually_excluded
            })
            if not is_baseline_start:
                active_idx += 1

        # Filter the fit dataset for actual Weibull fitting
        fit_data_for_fit = fit_data.drop(index=indices_to_drop)
        fit_data_for_fit = fit_data_for_fit[(fit_data_for_fit["TBX"] >= min_tbx) & (fit_data_for_fit["TBX"] > 0)].copy()

        fitter = ReliabilityFitter(fit_data_for_fit)
        results = fitter.fit_weibull(
            column=target_col, censored_failure_types=req.censored_failure_types
        )

        if not results or results.get("beta") is None:
            return {
                "status": "error",
                "message": "Could not fit Weibull distribution",
                "parameters": None,
                "intervals": intervals_list,
            }

        curve_data = None
        try:
            from reliability.Distributions import Weibull_Distribution

            if results.get("beta") and results.get("eta"):
                dist = Weibull_Distribution(alpha=results["eta"], beta=results["beta"])
                t_max = (
                    float(fit_data_for_fit["TBX"].max())
                    if "TBX" in fit_data_for_fit.columns
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
                "beta": float(results["beta"]) if results.get("beta") is not None else None,
                "eta": float(results["eta"]) if results.get("eta") is not None else None,
            },
            "goodness_of_fit": {
                "aic": float(results.get("aic")) if results.get("aic") is not None else None,
                "bic": float(results.get("bic")) if results.get("bic") is not None else None,
                "p_value": float(results.get("p_value")) if results.get("p_value") is not None else None,
                "ks_stat": float(results.get("ks_stat")) if results.get("ks_stat") is not None else None,
            },
            "mtbf": float(results.get("mtbf")) if results.get("mtbf") is not None else None,
            "reliability_curve": curve_data,
            "sample_size": len(fit_data_for_fit),
            "failures_count": results.get("failures_count"),
            "censored_count": results.get("censored_count"),
            "intervals": intervals_list,
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
        if req.types_to_use:
            data = data[data["Type"].isin(req.types_to_use)]
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
            failures_mtbf = int((group[uptime_col] > 0).sum()) if uptime_col else failures
            mtbf = uptime / failures_mtbf if failures_mtbf > 0 else 0.0
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
        if req.types_to_use:
            data = data[data["Type"].isin(req.types_to_use)]
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
async def kpi_trend(req: KpiTrendRequest) -> Dict[str, Any]:
    if state.current_data is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        df = state.current_data.copy()
        
        # Apply failure type filter if provided
        if req.failure_type:
            df = df[df["Type"] == req.failure_type]
            
        if req.types_to_use:
            df = df[df["Type"].isin(req.types_to_use)]
            
        if "Start_Date" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="Dataset must contain datetime start column (Start_Date).",
            )

        df = df.dropna(subset=["Start_Date"])
        if df.empty:
            return {"status": "success", "trend": [], "trends": {}, "months": []}

        df["month_period"] = df["Start_Date"].dt.to_period("M")
        all_months = sorted(df["month_period"].unique())

        # Determine target equipments
        available_eqs = sorted(df["Equipment"].dropna().unique().tolist())
        req_equipment = req.equipment
        if isinstance(req_equipment, str):
            req_equipment = [req_equipment] if req_equipment else []
        target_eqs = req_equipment if (req_equipment and len(req_equipment) > 0) else available_eqs

        # Calculate trends
        trends = {}
        
        # 1. Global (of the selected equipments)
        df_selected = df[df["Equipment"].isin(target_eqs)] if target_eqs else df
        grouped_global = df_selected.groupby("month_period")
        global_data = []
        for month in all_months:
            if month in grouped_global.groups:
                group = grouped_global.get_group(month)
                failures = int(len(group))
                downtime = float(group["TTX"].sum()) if "TTX" in group.columns else 0.0
                uptime = float(group["TBX"].sum()) if "TBX" in group.columns else 0.0
                failures_mtbf = int((group["TBX"] > 0).sum()) if "TBX" in group.columns else failures
                mtbf = float(uptime / failures_mtbf) if failures_mtbf > 0 else 0.0
                mttr = float(downtime / failures) if failures > 0 else 0.0
                total_time = uptime + downtime
                availability = float((uptime / total_time) * 100.0) if total_time > 0.0 else 0.0
            else:
                failures = 0
                downtime = 0.0
                uptime = 0.0
                mtbf = 0.0
                mttr = 0.0
                availability = 100.0
            global_data.append({
                "month": str(month),
                "failures": failures,
                "downtime": downtime,
                "mtbf": mtbf,
                "mttr": mttr,
                "availability": availability
            })
        trends["Global"] = global_data

        # 2. Per Equipment
        for eq in target_eqs:
            eq_df = df[df["Equipment"] == eq]
            grouped_eq = eq_df.groupby("month_period")
            eq_data = []
            for month in all_months:
                if month in grouped_eq.groups:
                    group = grouped_eq.get_group(month)
                    failures = int(len(group))
                    downtime = float(group["TTX"].sum()) if "TTX" in group.columns else 0.0
                    uptime = float(group["TBX"].sum()) if "TBX" in group.columns else 0.0
                    failures_mtbf = int((group["TBX"] > 0).sum()) if "TBX" in group.columns else failures
                    mtbf = float(uptime / failures_mtbf) if failures_mtbf > 0 else 0.0
                    mttr = float(downtime / failures) if failures > 0 else 0.0
                    total_time = uptime + downtime
                    availability = float((uptime / total_time) * 100.0) if total_time > 0.0 else 0.0
                else:
                    failures = 0
                    downtime = 0.0
                    uptime = 0.0
                    mtbf = 0.0
                    mttr = 0.0
                    availability = 100.0
                eq_data.append({
                    "month": str(month),
                    "failures": failures,
                    "downtime": downtime,
                    "mtbf": mtbf,
                    "mttr": mttr,
                    "availability": availability
                })
            trends[str(eq)] = eq_data

        return {
            "status": "success",
            "trend": global_data,
            "trends": trends,
            "months": [str(m) for m in all_months]
        }
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
            return {"status": "success", "results": {}}

        comment_col = None
        for col in data.columns:
            if col.lower() in ("comentario", "comment"):
                comment_col = col
                break

        if not comment_col:
            return {
                "status": "warning",
                "message": "No comment column found in dataset",
                "results": {},
            }

        comments_df = data.dropna(subset=[comment_col])
        valid_mask = ~comments_df[comment_col].astype(str).str.lower().isin(
            ["---", "nan", "none", "null", "no aplica", "n/a"]
        )
        valid_data = comments_df[valid_mask]

        total_records = len(data)
        comments_count = len(valid_data)
        coverage = (
            (comments_count / total_records * 100.0) if total_records > 0 else 0.0
        )

        if valid_data.empty:
            return {
                "status": "success",
                "results": {},
                "coverage": 0.0,
                "total_comments": 0,
            }

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

        def clean_word(w: str) -> str:
            w = w.lower()
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
            return glitches.get(w, w)

        def reduce_plural(w: str) -> str:
            if w.endswith("ces"):
                return w[:-3] + "z"
            if w.endswith("es") and len(w) > 4:
                c = w[:-2]
                if c[-1] in "bcdfghjklmnpqrstvwxyz":
                    return c
            if (
                w.endswith("s")
                and len(w) > 3
                and not w.endswith("is")
                and not w.endswith("us")
            ):
                return w[:-1]
            return w

        from collections import Counter
        import re

        all_terms = []
        texts_list = []

        for _, row in valid_data.iterrows():
            text = str(row[comment_col])
            texts_list.append(text)
            text_lower = text.lower()
            words = re.findall(r"\b[a-zA-Záéíóúñ]{3,}\b", text_lower)
            cleaned_words = [reduce_plural(clean_word(w)) for w in words]
            for w in cleaned_words:
                if w not in core_stop_words and w not in generic_words and len(w) > 2:
                    all_terms.append(w)
            for i in range(len(cleaned_words) - 1):
                w1, w2 = cleaned_words[i], cleaned_words[i + 1]
                if (
                    w1 not in core_stop_words
                    and w2 not in core_stop_words
                    and (w1 not in generic_words or w2 not in generic_words)
                ):
                    all_terms.append(f"{w1} {w2}")

        counter = Counter(all_terms)
        top_keywords = [
            {"word": word, "count": count} for word, count in counter.most_common(15)
        ]

        type_col = "Type" if "Type" in data.columns else None
        mode_col = "mdf"

        final_results = {}
        target_labels = [
            "Operational",
            "Cleaning/Blockage",
            "Mechanical",
            "Electrical",
            "Instrumentation/Failure",
            "Others",
        ]

        models_to_run = req.types_to_use if req.types_to_use else [NLP_MODELS_TO_COMPARE[0]]
        if "Todos los modelos" in models_to_run or "All" in models_to_run:
            models_to_run = NLP_MODELS_TO_COMPARE

        for model_name in models_to_run:
            start_time = time.time()
            analyzed_records = []

            if model_name == "Legacy Keyword NLP":
                for _, row in valid_data.iterrows():
                    text_lower = str(row[comment_col]).lower()
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
                            "calibration",
                            "instrumentation",
                            "instrument",
                        ]
                    ):
                        cat = "Instrumentation/Failure"

                    analyzed_records.append(
                        {
                            "category": cat,
                            "type": str(row.get(type_col, "Unknown")),
                            "mode": str(row.get(mode_col, "Unknown")),
                        }
                    )
            else:
                candidate_labels = [
                    "Operational",
                    "Cleaning or Blockage",
                    "Mechanical",
                    "Electrical",
                    "Instrumentation or Failure",
                    "Others",
                ]
                try:
                    predictions = SemanticModelManager.batch_predict(
                        texts_list, model_name, candidate_labels
                    )
                    label_map = {
                        "Cleaning or Blockage": "Cleaning/Blockage",
                        "Instrumentation or Failure": "Instrumentation/Failure",
                    }
                    for i, (_, row) in enumerate(valid_data.iterrows()):
                        raw_cat = predictions[i] if i < len(predictions) else "Others"
                        cat = label_map.get(raw_cat, raw_cat)
                        if cat not in target_labels:
                            cat = "Others"
                        analyzed_records.append(
                            {
                                "category": cat,
                                "type": str(row.get(type_col, "Unknown")),
                                "mode": str(row.get(mode_col, "Unknown")),
                            }
                        )
                except Exception as ex:
                    logger.error(f"Error running model {model_name}: {ex}")
                    for _, row in valid_data.iterrows():
                        analyzed_records.append(
                            {
                                "category": "Others",
                                "type": str(row.get(type_col, "Unknown")),
                                "mode": str(row.get(mode_col, "Unknown")),
                            }
                        )

            categories_details = []
            for cat_name in target_labels:
                cat_records = [r for r in analyzed_records if r["category"] == cat_name]
                count = len(cat_records)
                top_types = [
                    t
                    for t, _ in Counter([r["type"] for r in cat_records]).most_common(3)
                ]
                top_modes = [
                    m
                    for m, _ in Counter([r["mode"] for r in cat_records]).most_common(3)
                ]
                categories_details.append(
                    {
                        "category": cat_name,
                        "count": count,
                        "top_types": top_types,
                        "top_modes": top_modes,
                    }
                )

            execution_time = round(time.time() - start_time, 2)
            final_results[model_name] = {
                "categories": categories_details,
                "keywords": top_keywords,
                "execution_time_seconds": execution_time,
                "predictions": [r["category"] for r in analyzed_records],
            }

        return {
            "status": "success",
            "coverage": float(coverage),
            "total_comments": int(comments_count),
            "results": final_results,
        }
    except Exception as e:
        logger.error(f"Comment mining error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/analysis/models-status", tags=["Analysis"])
async def models_status() -> Dict[str, Any]:
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        cached_repo_sizes = {}
        for repo in cache_info.repos:
            cached_repo_sizes[repo.repo_id] = repo.size_on_disk / (1024 * 1024)
    except Exception as e:
        logger.warning(f"Could not scan huggingface cache: {e}")
        cached_repo_sizes = {}

    models = []
    from src.reliability_analysis.utils.config import NLP_MODELS_TO_COMPARE
    for model_name in NLP_MODELS_TO_COMPARE:
        if model_name == "Legacy Keyword NLP":
            downloaded = True
            size_mb = 0.0
        else:
            downloaded = model_name in cached_repo_sizes
            size_mb = cached_repo_sizes.get(model_name, 0.0)
            
        models.append({
            "name": model_name,
            "downloaded": downloaded,
            "size_mb": round(size_mb, 1)
        })
    return {"status": "success", "models": models}

@router.post("/analysis/download-model", tags=["Analysis"])
async def download_model(req: AnalysisRequest) -> Dict[str, Any]:
    from src.reliability_analysis.utils.config import NLP_MODELS_TO_COMPARE
    from src.reliability_analysis.analysis.hf_classifier import SemanticModelManager
    
    models_to_run = req.types_to_use if req.types_to_use else [NLP_MODELS_TO_COMPARE[0]]
    if "Todos los modelos" in models_to_run or "All" in models_to_run:
        models_to_run = NLP_MODELS_TO_COMPARE

    hf_models = [m for m in models_to_run if m != "Legacy Keyword NLP"]

    downloaded_models = []
    try:
        for model_name in hf_models:
            # Getting the pipeline forces the download if it is missing
            SemanticModelManager.get_pipeline(model_name)
            downloaded_models.append(model_name)
            
        return {"status": "success", "downloaded": downloaded_models}
    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analysis/kijima-fit", tags=["Analysis"])
async def kijima_fit(req: KijimaFitRequest) -> Dict[str, Any]:
    if state.current_data is None or state.filter_manager is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        data = state.filter_manager.get_filtered_data()

        if req.types_to_fit:
            all_types = list(req.types_to_fit)
            if req.censored_failure_types:
                for t in req.censored_failure_types:
                    if t not in all_types:
                        all_types.append(t)
            data = data[data["Type"].isin(all_types)]

        if data.empty:
            return {
                "status": "error",
                "message": "Filtered data is empty",
                "models": [],
            }

        fit_data = data.copy()
        target_col = "TBX"

        if "Days" in fit_data.columns and "TBX" not in fit_data.columns:
            target_col = "Days"

        if target_col in fit_data.columns and target_col != "TBX":
            fit_data["TBX"] = fit_data[target_col]

        if "TBX" not in fit_data.columns:
            raise HTTPException(status_code=400, detail="No time-between-failures (TBX) column found in the dataset.")

        censored_types = req.censored_failure_types or []
        min_tbx = float(req.min_tbx) if req.min_tbx is not None else 0.0
        excluded_idxs = req.excluded_indices or []

        # Build the intervals list for output
        intervals_df = fit_data[fit_data["TBX"] >= 0.0].copy()
        if "Start_Date" in intervals_df.columns:
            intervals_df = intervals_df.sort_values("Start_Date")
        
        intervals_list = []
        active_idx = 1
        indices_to_drop = []
        for i, (orig_idx, row) in enumerate(intervals_df.iterrows()):
            tbx_val = float(row["TBX"])
            is_baseline_start = (i == 0)
            
            is_manually_excluded = False
            if not is_baseline_start:
                is_manually_excluded = (active_idx in excluded_idxs)
                
            is_included = (tbx_val >= min_tbx) and (tbx_val > 0.0) and not is_baseline_start and not is_manually_excluded
            if is_manually_excluded:
                indices_to_drop.append(orig_idx)
            
            date_str = str(row["Start_Date"]) if "Start_Date" in row and pd.notnull(row["Start_Date"]) else "-"
            type_val = str(row["Type"]) if "Type" in row else "-"
            mode_val = str(row["mdf"]) if "mdf" in row else "-"
            
            intervals_list.append({
                "index": active_idx,
                "date": date_str,
                "tbx": tbx_val,
                "type": type_val,
                "mode": mode_val,
                "included": is_included,
                "is_baseline": is_baseline_start,
                "manually_excluded": is_manually_excluded
            })
            if not is_baseline_start:
                active_idx += 1

        # Filter the fit dataset for Kijima fitting (must have TBX >= min_tbx)
        fit_data_for_fit = fit_data.drop(index=indices_to_drop)
        fit_data_for_fit = fit_data_for_fit[(fit_data_for_fit["TBX"] >= min_tbx) & (fit_data_for_fit["TBX"] > 0)].copy()

        fitter = KijimaFitter()
        results = fitter.fit(
            dataframe=fit_data_for_fit,
            column="TBX",
            censored_types=censored_types,
            models=[1, 2, 3, 4, 5, 6],
        )

        if isinstance(results, dict):
            results = [results]

        # Append Weibull baseline model
        try:
            w_fitter = ReliabilityFitter(fit_data_for_fit)
            w_res = w_fitter.fit_weibull(column="TBX", censored_failure_types=censored_types)
            if w_res and "beta" in w_res and "eta" in w_res:
                w_beta = w_res["beta"]
                w_eta = w_res["eta"]

                from src.reliability_analysis.analysis.kijima_model import KijimaModelI
                w_model = KijimaModelI(w_beta, w_eta, 0.0, 0.0)

                df_prep = fit_data_for_fit.dropna(subset=["TBX", "Type" if "Type" in fit_data_for_fit.columns else "mdf"]).copy()
                df_prep = df_prep[df_prep["TBX"] > 0]
                x_arr = df_prep["TBX"].to_numpy(dtype=float)
                col_name = "Type" if "Type" in df_prep.columns else "mdf"
                delta_arr = (~df_prep[col_name].isin(censored_types)).astype(float).to_numpy()

                w_curves = w_model.calculate_curves(x_arr, delta_arr)

                weibull_baseline = {
                    "model_name": "Weibull",
                    "beta": w_beta,
                    "eta": w_eta,
                    "ar": 0.0,
                    "ap": 0.0,
                    "br": 0.0,
                    "bp": 0.0,
                    "AIC": w_res.get("aic"),
                    "BIC": w_res.get("bic"),
                    "p_value": w_res.get("p_value"),
                    "mean": w_res.get("mtbf"),
                    "ks_stat": w_res.get("ks_stat"),
                    "std": 0.0,
                }
                weibull_baseline.update(w_curves)
                results.append(weibull_baseline)
        except Exception as ex:
            logger.warning(f"Could not compute Weibull GRP baseline: {ex}")

        import math

        def sanitize_value(v):
            if isinstance(v, (float, np.floating)):
                v = float(v)
                if math.isnan(v) or math.isinf(v):
                    return None
                return v
            if isinstance(v, (int, np.integer)):
                return int(v)
            if isinstance(v, np.ndarray):
                return [sanitize_value(x) for x in v.tolist()]
            if isinstance(v, list):
                return [sanitize_value(x) for x in v]
            if isinstance(v, dict):
                return {k2: sanitize_value(v2) for k2, v2 in v.items()}
            return v

        serialized_results = []
        for res in results:
            serialized_results.append({k: sanitize_value(v) for k, v in res.items()})

        return {
            "status": "success",
            "models": serialized_results,
            "sample_size": len(fit_data_for_fit),
            "intervals": intervals_list,
        }
    except Exception as e:
        logger.error(f"Kijima fit analysis error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analysis/rcm/suggest", tags=["ISO Analysis"])
async def rcm_suggest(req: RcmSuggestRequest) -> Dict[str, Any]:
    """Generates RCM suggestions for the selected equipment based on SAE JA1011."""
    if state.filter_manager is None or state.current_data is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        df = state.current_data[state.current_data["Equipment"] == req.equipment].copy()
        comments = []
        if not df.empty and "Comment" in df.columns:
            comments = df["Comment"].dropna().astype(str).tolist()

        customized = LlmService.get_rcm_suggestions(req.equipment, comments)

        return {
            "status": "success",
            "equipment": req.equipment,
            "rcm_sheets": customized
        }
    except Exception as e:
        logger.error(f"RCM suggestion error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analysis/fmea/calculate-rpn", tags=["ISO Analysis"])
async def fmea_calculate_rpn(req: FmecaRpnRequest) -> Dict[str, Any]:
    """Calculates Risk Priority Number (RPN) and assigns risk categories according to IEC 60812."""
    rpn = req.severity * req.occurrence * req.detection
    
    if rpn < 50:
        category = "Bajo"
        color = "green"
    elif rpn < 150:
        category = "Medio"
        color = "yellow"
    elif rpn < 300:
        category = "Alto"
        color = "orange"
    else:
        category = "Crítico"
        color = "red"
        
    return {
        "status": "success",
        "rpn": rpn,
        "category": category,
        "color": color
    }


@router.post("/analysis/ram/simulate", tags=["ISO Analysis"])
async def ram_simulate(req: RamSimulateRequest) -> Dict[str, Any]:
    """Runs a plant Availability and Production Assurance simulation based on ISO 20815."""
    if state.filter_manager is None or state.current_data is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        # Base data filtering
        df = state.current_data.copy()
        if req.equipment:
            df = df[df["Equipment"] == req.equipment]
            
        if df.empty:
            raise ValueError("No data available for the selected equipment")

        # Total failures and total downtime
        num_failures = int(df["Type"].isin(["CORRECTIVO", "MI"]).sum())
        if num_failures == 0:
            num_failures = int(len(df))
            
        actual_downtime = float(df["TTX"].sum()) if "TTX" in df.columns else float(num_failures * 2.0)
        
        # Total simulation horizon (e.g. 1 year of operation: 8760 hours)
        horizon = 8760.0
        
        # Calculate simulated downtime adjusting for efficiency and logistics delay
        # More logistics delay increases downtime; higher preventive efficiency reduces failures/downtime
        logistics_factor = req.logistics_delay * num_failures
        preventive_reduction = 1.0 - (req.preventive_efficiency * 0.4)
        
        simulated_downtime = (actual_downtime + logistics_factor) * preventive_reduction
        simulated_downtime = min(horizon - 100.0, max(1.0, simulated_downtime))
        
        simulated_uptime = horizon - simulated_downtime
        availability = (simulated_uptime / horizon) * 100.0
        production_assurance = availability * 0.985 # Subtract minor processing losses
        
        # Generate monthly timeline data for availability chart
        monthly_availability = []
        months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        
        # Add random fluctuations around the average availability
        np.random.seed(42)
        noise = np.random.normal(0, 1.5, 12)
        for i, month in enumerate(months):
            val = min(100.0, max(50.0, availability + noise[i]))
            monthly_availability.append({"month": month, "availability": round(val, 2)})

        # Downtime contributors (Bad Actors)
        equipments = state.current_data["Equipment"].unique()
        bad_actors_contrib = []
        for eq in equipments:
            eq_df = state.current_data[state.current_data["Equipment"] == eq]
            eq_downtime = eq_df["TTX"].sum() if "TTX" in eq_df.columns else len(eq_df) * 2.0
            bad_actors_contrib.append({
                "equipment": eq,
                "downtime": round(float(eq_downtime), 1),
                "failures": int(len(eq_df))
            })
        bad_actors_contrib = sorted(bad_actors_contrib, key=lambda x: x["downtime"], reverse=True)[:5]

        return {
            "status": "success",
            "availability": round(availability, 2),
            "production_assurance": round(production_assurance, 2),
            "uptime_hours": round(simulated_uptime, 1),
            "downtime_hours": round(simulated_downtime, 1),
            "bad_actors": bad_actors_contrib,
            "timeline": monthly_availability
        }
    except Exception as e:
        logger.error(f"RAM simulation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analysis/rca/suggest", tags=["ISO Analysis"])
async def rca_suggest(req: RcaAnalysisRequest) -> Dict[str, Any]:
    """Generates a Root Cause Analysis (Ishikawa & 5 Whys) suggestion based on IEC 62740."""
    if state.filter_manager is None or state.current_data is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        df = state.current_data[state.current_data["Equipment"] == req.equipment].copy()
        comments = []
        if not df.empty and "Comment" in df.columns:
            comments = df["Comment"].dropna().astype(str).tolist()

        result = LlmService.get_rca_suggestions(req.equipment, comments)

        return {
            "status": "success",
            "equipment": req.equipment,
            "five_whys": result.get("five_whys", []),
            "ishikawa": result.get("ishikawa", {})
        }
    except Exception as e:
        logger.error(f"RCA suggestion error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


