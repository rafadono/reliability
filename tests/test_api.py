"""
Unit tests for API endpoints.

Tests all API endpoints for data upload, filtering, and analysis.
"""

import pytest
import io
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client for API."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
    from app import app

    return TestClient(app)


@pytest.fixture
def test_csv():
    """Create test CSV file."""
    csv_content = """Equipo;Tipo;Mdf;Dias;Censurado;Fecha;Comentario
Motor A;Mechanical;Bearing;100;0;01/01/2026;Falla mecanica de rodamiento debido a desgaste
Motor A;Mechanical;Shaft;150;0;15/01/2026;Shaft roto por vibracion
Pump B;Hydraulic;Seal;120;1;01/02/2026;Falla por decision operacional
Motor A;Electrical;Coil;180;0;10/02/2026;Bobina quemada y daño electrico
Pump B;Mechanical;Bearing;200;0;20/02/2026;Limpieza por atollo de circuito"""
    return io.BytesIO(csv_content.encode())


class TestAPIEndpoints:
    """Test suite for API endpoints."""

    @pytest.fixture(autouse=True)
    def setup_data(self, client, test_csv):
        """Upload data before each test in this class."""
        files = {"file": ("test.csv", test_csv, "text/csv")}
        response = client.post("/api/upload", files=files)
        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_health_check(self, client: TestClient):
        """Test API is running."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["data_loaded"] is True

    def test_get_available_filters(self, client: TestClient):
        """Test get all available filters endpoint."""
        response = client.get("/api/data/available-filters")
        assert response.status_code == 200
        data = response.json()
        assert "equipment" in data
        assert "types" in data
        assert "Motor A" in data["equipment"]

    def test_set_filters(self, client: TestClient):
        """Test setting filters."""
        response = client.post("/api/filters/set", json={"equipment": "Motor A"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["rows_filtered"] > 0

    def test_pareto_analysis(self, client: TestClient):
        """Test Pareto analysis endpoint."""
        response = client.post("/api/analysis/pareto", json={"group_by": "equipo"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "pareto" in data
        assert len(data["pareto"]) > 0

    def test_jackknife_plot_analysis(self, client: TestClient):
        """Test Jackknife plot analysis endpoint."""
        response = client.post(
            "/api/analysis/jackknife-plot", json={"compare_by": "equipment"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "scatter_data" in data

    def test_criticality_plot_analysis(self, client: TestClient):
        """Test Criticality plot analysis endpoint."""
        response = client.post(
            "/api/analysis/criticality-plot", json={"compare_by": "mode", "metric_x": "count"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "scatter_data" in data
        assert "regions" in data
        assert "highRisk" in data["regions"]

    def test_summary_stats(self, client: TestClient):
        """Test summary statistics endpoint."""
        response = client.get("/api/stats/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "total_records" in data

    def test_cors_headers(self, client: TestClient):
        """Test CORS headers are set."""
        response = client.get(
            "/api/data/available-filters", headers={"Origin": "http://localhost"}
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] in ["http://localhost", "*"]

    def test_kpi_trend_analysis(self, client: TestClient):
        """Test KPI Trend analysis endpoint."""
        response = client.post("/api/analysis/kpi-trend", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "trend" in data
        assert len(data["trend"]) > 0
        assert "month" in data["trend"][0]
        assert "failures" in data["trend"][0]
        assert "mtbf" in data["trend"][0]
        assert "availability" in data["trend"][0]

    def test_comment_mining_analysis(self, client: TestClient):
        """Test Comment Mining NLP endpoint."""
        response = client.post("/api/analysis/comment-mining", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "coverage" in data
        assert "results" in data
        assert "Legacy Keyword NLP" in data["results"]
        
        legacy_results = data["results"]["Legacy Keyword NLP"]
        assert "keywords" in legacy_results
        assert "categories" in legacy_results
        assert "execution_time_seconds" in legacy_results
        
        if len(legacy_results["keywords"]) > 0:
            assert "word" in legacy_results["keywords"][0]
            assert "count" in legacy_results["keywords"][0]
        
        if len(legacy_results["categories"]) > 0:
            assert "category" in legacy_results["categories"][0]
            assert "top_types" in legacy_results["categories"][0]
            assert "top_modes" in legacy_results["categories"][0]

    def test_fit_analysis(self, client: TestClient):
        """Test Weibull Fit endpoint exposes MTBF & KS metrics."""
        response = client.post(
            "/api/analysis/fit",
            json={
                "target_column": "Days",
                "censored_failure_types": []
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "parameters" in data
        assert "beta" in data["parameters"]
        assert "eta" in data["parameters"]
        assert "goodness_of_fit" in data
        assert "p_value" in data["goodness_of_fit"]
        assert "ks_stat" in data["goodness_of_fit"]
        assert "mtbf" in data
        assert data["mtbf"] is not None
