from fastapi import APIRouter, File, UploadFile, HTTPException
import pandas as pd
import io
import logging
import traceback
from typing import List, Dict

from models.requests import UploadResponse
import state
from src.reliability_analysis.core.data_processing import DataProcessor
from src.reliability_analysis.core.filters import FilterManager

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=UploadResponse, tags=["Data"])
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload CSV file with reliability data.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), sep=";")

        state.data_processor = DataProcessor()
        state.current_data = state.data_processor.treat_data(df)
        state.filter_manager = FilterManager(state.current_data)

        logger.info(
            f"Uploaded {len(state.current_data)} records with columns: {state.current_data.columns.tolist()}"
        )

        return UploadResponse(
            status="success",
            rows_loaded=len(state.current_data),
            columns=state.current_data.columns.tolist(),
            message=f"Successfully loaded {len(state.current_data)} records",
        )
    except Exception as e:
        logger.error(f"Upload error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/data/available-filters", tags=["Data"])
async def get_available_filters() -> Dict[str, List[str]]:
    """Get all available equipment, types, and modes (no filters applied)."""
    if state.current_data is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    try:
        return {
            "equipment": sorted(state.current_data["Equipment"].unique().tolist()),
            "types": sorted(state.current_data["Type"].unique().tolist()),
            "failure_modes": sorted(state.current_data["mdf"].unique().tolist()),
        }
    except Exception as e:
        logger.error(f"Get available filters error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
