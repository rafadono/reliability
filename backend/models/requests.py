from pydantic import BaseModel, Field
from typing import Optional, List, Union


class UploadResponse(BaseModel):
    status: str
    rows_loaded: int
    columns: List[str]
    message: str


class FilterOptions(BaseModel):
    equipment: Optional[List[str]] = None
    types: Optional[List[str]] = None
    failure_modes: Optional[List[str]] = None


class FilterRequest(BaseModel):
    equipment: Optional[str] = Field(None, description="Equipment name")
    failure_type: Optional[str] = Field(None, description="Failure type")
    failure_mode: Optional[str] = Field(None, description="Failure mode")


class ParetoRequest(BaseModel):
    group_by: str = Field(..., description="'equipo', 'tipo', or 'mdf'")
    equipment: Optional[str] = None
    failure_type: Optional[str] = None


class AnalysisRequest(BaseModel):
    equipment: Optional[str] = None
    failure_type: Optional[str] = None
    failure_mode: Optional[str] = None
    types_to_use: Optional[List[str]] = Field(
        None, description="List of types to include in the analysis"
    )
    compare_by: str = Field("equipment", description="Compare by 'equipment' or 'type'")


class WeibullFitRequest(BaseModel):
    equipment: Optional[str] = None
    failure_type: Optional[str] = None
    failure_mode: Optional[str] = None
    censored_failure_types: Optional[List[str]] = Field(
        None, description="List of failure types to consider as censored"
    )
    types_to_fit: Optional[List[str]] = Field(
        None, description="List of failure types to fit"
    )
    target_column: Optional[str] = Field(
        "TBX",
        description="Column to fit (TBX for Reliability, TTX for Maintainability)",
    )
    min_tbx: Optional[float] = Field(
        0.0,
        description="Minimum TBX (hours) to include in fit"
    )
    excluded_indices: Optional[List[int]] = Field(
        None,
        description="List of 1-based indices of intervals to exclude from fit"
    )


class OptimalPMRequest(WeibullFitRequest):
    cost_pm: float = Field(
        ..., description="Cost of a single Preventive Maintenance action"
    )
    cost_failure: float = Field(
        ..., description="Cost of a single Corrective Maintenance action (failure)"
    )


class ConditionalReliabilityRequest(WeibullFitRequest):
    current_age: float = Field(
        ..., description="Current age of the component since last major overhaul or PM"
    )
    mission_time: float = Field(
        ..., description="The duration of the future mission to evaluate"
    )


class CriticalityRequest(AnalysisRequest):
    metric_x: str = Field("count", description="X-axis metric: 'count' or 'probability'")


class KijimaFitRequest(BaseModel):
    equipment: Optional[str] = None
    failure_type: Optional[str] = None
    failure_mode: Optional[str] = None
    censored_failure_types: Optional[List[str]] = Field(
        None, description="List of failure types to consider as censored (preventive)"
    )
    types_to_fit: Optional[List[str]] = Field(
        None, description="List of failure types to fit"
    )
    min_tbx: Optional[float] = Field(
        0.0,
        description="Minimum TBX (hours) to include in fit"
    )
    excluded_indices: Optional[List[int]] = Field(
        None,
        description="List of 1-based indices of intervals to exclude from fit"
    )


class KpiTrendRequest(BaseModel):
    equipment: Optional[Union[str, List[str]]] = None
    failure_type: Optional[str] = None
    types_to_use: Optional[List[str]] = Field(
        None, description="List of types to include in the analysis"
    )


class RcmSuggestRequest(BaseModel):
    equipment: str = Field(..., description="Name of the equipment unit")


class FmecaRpnRequest(BaseModel):
    severity: int = Field(..., ge=1, le=10, description="Severity rating (1-10)")
    occurrence: int = Field(..., ge=1, le=10, description="Occurrence rating (1-10)")
    detection: int = Field(..., ge=1, le=10, description="Detection rating (1-10)")


class RamSimulateRequest(BaseModel):
    equipment: Optional[str] = Field(None, description="Equipment to simulate. If None, runs global plant simulation.")
    preventive_efficiency: float = Field(0.8, ge=0.0, le=1.0, description="Efficiency of preventive maintenance (0 to 1)")
    logistics_delay: float = Field(4.0, ge=0.0, description="Average logistics delay in hours")


class RcaAnalysisRequest(BaseModel):
    equipment: str = Field(..., description="Equipment name")
    failure_event_date: Optional[str] = Field(None, description="Optional date of the failure event to analyze")

