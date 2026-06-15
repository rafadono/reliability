import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.reliability_analysis.analysis.kijima_model import (
    calculate_ki,
    calculate_ki_td,
    calculate_k2,
    calculate_k2_td,
    calculate_virtual_age,
)
from src.reliability_analysis.analysis.models import KijimaFitter


def test_kijima_td_zero_slope_matches_constant():
    """
    Verifies that Kijima I/II Time-Dependent equations match
    the Constant ones when slopes are zero.
    """
    x = np.array([10.0, 25.0, 40.0, 15.0])
    delta = np.array([1.0, 0.0, 1.0, 1.0])
    ar, ap = 0.4, 0.8
    
    # Kijima I
    v_c1 = calculate_ki(x, delta, ar, ap)
    v_td1 = calculate_ki_td(x, delta, ar, ap, 0.0, 0.0)
    assert np.allclose(v_c1, v_td1)

    # Kijima II
    v_c2 = calculate_k2(x, delta, ar, ap)
    v_td2 = calculate_k2_td(x, delta, ar, ap, 0.0, 0.0)
    assert np.allclose(v_c2, v_td2)


def test_kijima_td_clipping():
    """
    Verifies that the restoration factor q = q0 + b*T is properly clipped to [0, 1].
    """
    x = np.array([10.0, 1000.0])
    delta = np.array([1.0, 1.0])
    
    # Case A: extreme positive slope, should clip at 1.0
    # T_0 = 10, q = 0.5 + 0.1 * 10 = 1.5 -> clipped to 1.0
    # T_1 = 1010, q = 0.5 + 0.1 * 1010 = 101.5 -> clipped to 1.0
    v_td = calculate_ki_td(x, delta, 0.5, 0.5, 0.1, 0.1)
    
    # Expected under q=1.0:
    # i=0: V_0 = 0 + 1.0 * 10 = 10
    # i=1: V_1 = 10 + 1.0 * 1000 = 1010
    assert np.allclose(v_td, [10.0, 1010.0])

    # Case B: extreme negative slope, should clip at 0.0
    # T_0 = 10, q = 0.5 - 0.1 * 10 = -0.5 -> clipped to 0.0
    # T_1 = 1010, q = 0.5 - 0.1 * 1010 = -100.5 -> clipped to 0.0
    v_td_neg = calculate_ki_td(x, delta, 0.5, 0.5, -0.1, -0.1)
    assert np.allclose(v_td_neg, [0.0, 0.0])


def test_calculate_virtual_age_dispatcher():
    """
    Verifies that the calculate_virtual_age dispatcher handles model types [1, 2, 3, 4].
    """
    x = np.array([10.0, 20.0])
    delta = np.array([1.0, 0.0])
    
    v1 = calculate_virtual_age(x, delta, 0.5, 0.2, 1)
    assert len(v1) == 2
    
    v2 = calculate_virtual_age(x, delta, 0.5, 0.2, 2)
    assert len(v2) == 2

    v3 = calculate_virtual_age(x, delta, 0.5, 0.2, 3, 0.01, -0.01)
    assert len(v3) == 2

    v4 = calculate_virtual_age(x, delta, 0.5, 0.2, 4, 0.01, -0.01)
    assert len(v4) == 2


def test_kijima_fitter_runs_and_contains_expected_output_structure():
    """
    Verifies that KijimaFitter runs and returns correct keys and structures.
    """
    import pandas as pd
    np.random.seed(42)
    # Create artificial dataset
    n = 30
    tbx = np.random.exponential(100.0, n) + 1.0
    mdf = np.random.choice(["Mechanical", "Preventive", "Corrective"], n)
    df = pd.DataFrame({"TBX": tbx, "mdf": mdf})

    fitter = KijimaFitter()
    results = fitter.fit(df, column="TBX", censored_types=["Preventive"], models=[1, 2, 3, 4])
    
    assert isinstance(results, list)
    assert len(results) == 4
    
    for res in results:
        assert "model_name" in res
        assert "beta" in res
        assert "eta" in res
        assert "ar" in res
        assert "ap" in res
        assert "br" in res
        assert "bp" in res
        assert "AIC" in res
        assert "BIC" in res
        assert "p_value" in res
        assert "mean" in res
        assert "ks_stat" in res
        assert "std" in res
        assert "t" in res
        assert "R" in res
        assert "failure_rate" in res
        assert "pdf" in res
        assert "V" in res
        assert "T" in res
        
        # Verify sizes
        assert len(res["t"]) == 300
        assert len(res["R"]) == 300
        assert len(res["failure_rate"]) == 300
        assert len(res["pdf"]) == 300
        assert len(res["V"]) == len(tbx)
        assert len(res["T"]) == len(tbx) + 1


def test_kijima_fit_api_endpoint():
    """
    Tests the new POST /api/analysis/kijima-fit API endpoint.
    """
    import sys
    from pathlib import Path
    import io
    
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
    from app import app
    
    client = TestClient(app)
    
    # Upload sample CSV first
    csv_content = """Equipo;Tipo;Mdf;Dias;Censurado;Fecha;Comentario
Motor A;Mechanical;Bearing;100;0;01/01/2026;Falla mecanica
Motor A;Mechanical;Shaft;150;0;15/01/2026;Shaft roto
Pump B;Hydraulic;Preventive;120;1;01/02/2026;PM preventivo
Motor A;Electrical;Coil;180;0;10/02/2026;Bobina quemada
Pump B;Mechanical;Bearing;200;0;20/02/2026;Falla rodamiento
Motor A;Mechanical;Seal;90;0;25/02/2026;Falla sello
"""
    files = {"file": ("test_kijima.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    upload_res = client.post("/api/upload", files=files)
    assert upload_res.status_code == 200
    
    # Now run Kijima Fit API
    fit_payload = {
        "censored_failure_types": ["Preventive"]
    }
    response = client.post("/api/analysis/kijima-fit", json=fit_payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert "models" in data
    assert len(data["models"]) == 4
    
    model_names = [m["model_name"] for m in data["models"]]
    assert "Kijima I" in model_names
    assert "Kijima II" in model_names
    assert "Kijima I TD" in model_names
    assert "Kijima II TD" in model_names
