"""
Pytest configuration and fixtures.
"""

import matplotlib

matplotlib.use("Agg")
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "backend"))
sys.path.insert(0, str(project_root / "backend" / "src"))


@pytest.fixture
def sample_data():
    """Sample reliability dataset for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Equipment": ["Motor A"] * 30 + ["Pump B"] * 25 + ["Motor C"] * 20,
            "Type": (
                ["Mechanical"] * 15
                + ["Electrical"] * 15
                + ["Hydraulic"] * 10
                + ["Mechanical"] * 15
                + ["Electrical"] * 10
                + ["Mechanical"] * 10
            ),
            "mdf": (
                ["Bearing"] * 8
                + ["Shaft"] * 7
                + ["Coil"] * 8
                + ["Winding"] * 7
                + ["Seal"] * 5
                + ["Pump"] * 5
                + ["Bearing"] * 8
                + ["Seal"] * 7
                + ["Coil"] * 5
                + ["Bearing"] * 5
                + ["Bearing"] * 10
            ),
            "Days": np.random.exponential(scale=150, size=75).astype(int) + 50,
            "Censored": [0] * 70 + [1] * 5,
        }
    )


@pytest.fixture
def small_data():
    """Small dataset for quick tests."""
    return pd.DataFrame(
        {
            "Equipment": ["Motor A"] * 10 + ["Pump B"] * 10,
            "Type": ["Mechanical"] * 10 + ["Hydraulic"] * 10,
            "mdf": ["Bearing"] * 5 + ["Shaft"] * 5 + ["Seal"] * 10,
            "Days": [
                100,
                150,
                200,
                120,
                180,
                90,
                110,
                140,
                170,
                160,
                85,
                95,
                110,
                130,
                145,
                75,
                105,
                125,
                135,
                155,
            ],
            "Censored": [0] * 18 + [1] * 2,
        }
    )
