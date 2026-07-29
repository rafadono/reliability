from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
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
    Upload CSV file with reliability data supporting multiple encodings and separators.
    """
    try:
        contents = await file.read()
        
        # Multi-encoding decode fallback
        contents_str = None
        for enc in ["utf-8-sig", "utf-8", "latin1", "cp1252"]:
            try:
                contents_str = contents.decode(enc)
                break
            except (UnicodeDecodeError, AttributeError):
                continue
                
        if contents_str is None:
            contents_str = contents.decode("utf-8", errors="replace")

        # Multi-separator fallback logic
        df = None
        for sep in [";", ",", "\t"]:
            try:
                temp_df = pd.read_csv(io.StringIO(contents_str), sep=sep)
                if len(temp_df.columns) > 1:
                    df = temp_df
                    break
            except Exception:
                continue
                
        if df is None:
            df = pd.read_csv(io.StringIO(contents_str), sep=None, engine="python")

        state.data_processor = DataProcessor()
        state.current_data = state.data_processor.treat_data(df)
        state.filter_manager = FilterManager(state.current_data)

        quality_report = state.current_data.attrs.get("quality_report", {})
        invalid_dates_count = quality_report.get("invalid_dates_count", 0)
        duplicates_removed_count = quality_report.get("duplicates_removed_count", 0)

        logger.info(
            f"Uploaded {len(state.current_data)} records with columns: {state.current_data.columns.tolist()}"
        )

        return UploadResponse(
            status="success",
            rows_loaded=len(state.current_data),
            columns=state.current_data.columns.tolist(),
            message=f"Successfully loaded {len(state.current_data)} records",
            invalid_dates_count=invalid_dates_count,
            duplicates_removed_count=duplicates_removed_count,
        )
    except Exception as e:
        logger.error(f"Upload error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/data/export", tags=["Data"])
async def export_data() -> StreamingResponse:
    """Exports the currently loaded (cleaned) dataset as a downloadable CSV."""
    if state.current_data is None:
        raise HTTPException(status_code=400, detail="No data loaded")

    output = io.StringIO()
    state.current_data.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=reliability_dataset.csv"},
    )


@router.get("/data/available-filters", tags=["Data"])
async def get_available_filters() -> Dict[str, List[str]]:
    """Get all available equipment, types, and modes (no filters applied)."""
    if state.current_data is None:
        return {
            "equipment": [],
            "types": [],
            "failure_modes": [],
        }

    try:
        eq_list = [str(x) for x in state.current_data["Equipment"].dropna().unique().tolist()] if "Equipment" in state.current_data.columns else []
        type_list = [str(x) for x in state.current_data["Type"].dropna().unique().tolist()] if "Type" in state.current_data.columns else []
        mdf_list = [str(x) for x in state.current_data["mdf"].dropna().unique().tolist()] if "mdf" in state.current_data.columns else []

        return {
            "equipment": sorted(eq_list),
            "types": sorted(type_list),
            "failure_modes": sorted(mdf_list),
        }
    except Exception as e:
        logger.error(f"Get available filters error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

