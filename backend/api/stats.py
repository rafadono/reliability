from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

import state

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/stats/summary", tags=["Statistics"])
async def get_summary_stats() -> Dict[str, Any]:
    """
    Get summary statistics for current filtered data.
    """
    if state.current_data is None or state.filter_manager is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        data = state.filter_manager.get_filtered_data()
        
        if data.empty:
            return {"status": "warning", "message": "No data in current filters"}
        
        return {
            "status": "success",
            "total_records": int(len(data)),
            "unique_equipment": int(data['Equipment'].nunique()),
            "unique_types": int(data['Type'].nunique()),
            "unique_modes": int(data['mdf'].nunique()),
            "ttx_min": float(data['TTX'].min()),
            "ttx_max": float(data['TTX'].max()),
            "ttx_mean": float(data['TTX'].mean()),
            "ttx_median": float(data['TTX'].median()),
            "ttx_std": float(data['TTX'].std())
        }
    except Exception as e:
        logger.error(f"Summary stats error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))