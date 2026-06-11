from pydantic import BaseModel, Field
from typing import Optional, List


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
