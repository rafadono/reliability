"""
FastAPI Backend for Reliability Analysis.

This is the main application file that orchestrates the API.
It initializes the FastAPI app, sets up middleware, and includes the routers
for different parts of the API (data, filters, analysis, stats).

Run: uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from pathlib import Path
import sys

# Add project root to path to allow imports from src
sys.path.insert(0, str(Path(__file__).parent))

# Import routers
from api import data, filters, analysis, stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Silence repetitive initialization logs to avoid console spam
# (especially useful during thousands of Jackknife iterations)
logging.getLogger("Models").setLevel(logging.WARNING)
logging.getLogger("FilterManager").propagate = False

app = FastAPI(
    title="Reliability Analysis API",
    description="Backend API for advanced reliability analysis with Pareto and Jackknife",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router, prefix="/api")
app.include_router(filters.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(stats.router, prefix="/api")

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    from state import current_data
    return {
        "status": "ok",
        "data_loaded": current_data is not None,
        "api_version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Reliability Analysis API...")
    print("Docs available at: http://localhost:8000/docs")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
