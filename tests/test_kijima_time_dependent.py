import numpy as np
import pytest
from fastapi.testclient import TestClient
from src.reliability_analysis.analysis.kijima_model import (
    KijimaModelI,
    KijimaModelII,
    KijimaModelITD,
    KijimaModelIITD,
)
from src.reliability_analysis.analysis.models import KijimaFitter


def test_kijima_td_zero_slope_matches_constant():
    """
    With br=bp=0, TD models must produce the same virtual age as constant models.
    """
    x = np.array([10.0, 25.0, 40.0, 15.0])
    delta = np.array([1.0, 0.0, 1.0, 1.0])
    ar, ap = 0.4, 0.8

    v_c1 = KijimaModelI(1.5, 500.0, ar, ap).virtual_age(x, delta)
    v_td1 = KijimaModelITD(1.5, 500.0, ar, ap, 0.0, 0.0).virtual_age(x, delta)
    np.testing.assert_allclose(v_c1, v_td1)

    v_c2 = KijimaModelII(1.5, 500.0, ar, ap).virtual_age(x, delta)
    v_td2 = KijimaModelIITD(1.5, 500.0, ar, ap, 0.0, 0.0).virtual_age(x, delta)
    np.testing.assert_allclose(v_c2, v_td2)


def test_kijima_td_clipping_upper():
    """Restoration factor q under exponential formulation for TD models."""
    x = np.array([10.0, 1000.0])
    delta = np.array([1.0, 1.0])

    v_td = KijimaModelITD(1.5, 500.0, 0.5, 0.5, 0.1, 0.1).virtual_age(x, delta)
    np.testing.assert_allclose(v_td, [8.160602794142788, 1008.1606027941427])


def test_kijima_td_clipping_lower():
    """Restoration factor q clipped at 0.0 when slope is very large negative."""
    x = np.array([10.0, 1000.0])
    delta = np.array([1.0, 1.0])

    v_td = KijimaModelITD(1.5, 500.0, 0.5, 0.5, -0.1, -0.1).virtual_age(x, delta)
    # q clips to 0.0 for both events -> all V_i = 0
    np.testing.assert_allclose(v_td, [0.0, 0.0])


def test_kijima_all_model_types_virtual_age():
    """All four model types produce finite, non-negative virtual ages."""
    x = np.array([10.0, 20.0])
    delta = np.array([1.0, 0.0])

    models = [
        KijimaModelI(1.5, 500.0, 0.5, 0.2),
        KijimaModelII(1.5, 500.0, 0.5, 0.2),
        KijimaModelITD(1.5, 500.0, 0.5, 0.2, 0.01, -0.01),
        KijimaModelIITD(1.5, 500.0, 0.5, 0.2, 0.01, -0.01),
    ]
    for model in models:
        V = model.virtual_age(x, delta)
        assert len(V) == 2
        assert np.all(np.isfinite(V))
        assert np.all(V >= 0.0)


def test_kijima_fitter_runs_and_contains_expected_output_structure():
    """KijimaFitter.fit() returns correct keys and array shapes."""
    import pandas as pd
    np.random.seed(42)
    n = 30
    tbx = np.random.exponential(100.0, n) + 1.0
    mdf = np.random.choice(["Mechanical", "Preventive", "Corrective"], n)
    df = pd.DataFrame({"TBX": tbx, "mdf": mdf})

    fitter = KijimaFitter()
    results = fitter.fit(df, column="TBX", censored_types=["Preventive"], models=[1, 2, 3, 4])

    assert isinstance(results, list)
    assert len(results) == 4

    for res in results:
        for key in ("model_name", "beta", "eta", "ar", "ap", "br", "bp",
                    "AIC", "BIC", "p_value", "mean", "ks_stat", "std",
                    "t", "R", "failure_rate", "pdf", "V", "T"):
            assert key in res

        assert len(res["t"]) == len(tbx) * 5 + 50
        assert len(res["R"]) == len(tbx) * 5 + 50
        assert len(res["V"]) == len(tbx)
        assert len(res["T"]) == len(tbx) + 1


def test_kijima_fit_api_endpoint():
    """POST /api/analysis/kijima-fit returns success with all five model names."""
    import sys
    import io
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
    from app import app

    client = TestClient(app)

    csv_content = (
        "Equipo;Tipo;Mdf;Dias;Censurado;Fecha;Comentario\n"
        "Motor A;Mechanical;Bearing;100;0;01/01/2026;Falla mecanica\n"
        "Motor A;Mechanical;Shaft;150;0;15/01/2026;Shaft roto\n"
        "Pump B;Hydraulic;Preventive;120;1;01/02/2026;PM preventivo\n"
        "Motor A;Electrical;Coil;180;0;10/02/2026;Bobina quemada\n"
        "Pump B;Mechanical;Bearing;200;0;20/02/2026;Falla rodamiento\n"
        "Motor A;Mechanical;Seal;90;0;25/02/2026;Falla sello\n"
    )
    files = {"file": ("test_kijima.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    assert client.post("/api/upload", files=files).status_code == 200

    response = client.post("/api/analysis/kijima-fit", json={"censored_failure_types": ["Preventive"]})
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert "models" in data
    assert len(data["models"]) == 7
    model_names = [m["model_name"] for m in data["models"]]
    for name in ("Kijima I", "Kijima II", "Kijima I TD", "Kijima II TD", "Kijima I TD2 (Logistic)", "Kijima II TD2 (Logistic)", "Weibull"):
        assert name in model_names
