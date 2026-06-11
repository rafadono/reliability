from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any

from models.requests import FilterOptions, FilterRequest
import state
from src.reliability_analysis.core.filters import FilterManager
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/filters", tags=["Filters"])
async def get_filters(
    equipment: Optional[str] = Query(None),
    failure_type: Optional[str] = Query(None)
) -> FilterOptions:
    """
    Get available filter options respecting hierarchy.
    """
    if state.filter_manager is None:
        raise HTTPException(status_code=400, detail="No data loaded. Upload a file first.")
    
    try:
        current_state = state.filter_manager.get_state()
        
        temp_fm = FilterManager(state.current_data)
        if equipment:
            temp_fm.set_equipment([equipment])
        if failure_type:
            temp_fm.set_types([failure_type])
        
        opts = FilterOptions(
            equipment=temp_fm.get_available_equipment(),
            types=temp_fm.get_available_types(),
            failure_modes=temp_fm.get_available_failure_modes()
        )
        
        return opts
    except Exception as e:
        logger.error(f"Filter error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/filters/set", tags=["Filters"])
async def set_filters(req: FilterRequest) -> Dict[str, Any]:
    """
    Apply filters with hierarchical cascade.
    """
    if state.filter_manager is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        state.filter_manager.reset()
        if req.equipment:
            state.filter_manager.set_equipment([req.equipment])
        if req.failure_type:
            state.filter_manager.set_types([req.failure_type])
        if req.failure_mode:
            state.filter_manager.set_failure_modes([req.failure_mode])
        
        filtered = state.filter_manager.get_filtered_data()
        
        return {
            "status": "success",
            "rows_filtered": len(filtered),
            "filters_applied": {
                "equipment": req.equipment,
                "failure_type": req.failure_type,
                "failure_mode": req.failure_mode
            }
        }
    except Exception as e:
        logger.error(f"Filter set error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/data/reset-filters", tags=["Filters"])
async def reset_filters() -> Dict[str, str]:
    """Reset all filters to show all data."""
    if state.filter_manager is None or state.current_data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        state.filter_manager = FilterManager(state.current_data)
        return {"status": "success", "message": "Filters reset"}
    except Exception as e:
        logger.error(f"Reset filters error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))