from fastapi.testclient import TestClient


class TestWorkbenchTemplates:
    """Test suite executing all 5 predefined DAG pipeline templates in code."""

    def test_template_basic_weibull(self, client: TestClient):
        """Test Basic Weibull Template: DataSource -> Filter -> Weibull 2P"""
        payload = {
            "nodes": [
                {"id": "source-1", "type": "dataSource", "data": {}, "x": 50, "y": 150},
                {
                    "id": "filter-1",
                    "type": "filter",
                    "data": {"equipment": "Motor A"},
                    "x": 280,
                    "y": 150,
                },
                {"id": "weibull-1", "type": "weibull", "data": {}, "x": 520, "y": 150},
            ],
            "edges": [
                {"id": "edge-1", "source": "source-1", "target": "filter-1"},
                {"id": "edge-2", "source": "filter-1", "target": "weibull-1"},
            ],
        }

        response = client.post("/api/workbench/execute", json=payload)
        assert response.status_code == 200
        res = response.json()
        assert res["status"] == "success"
        results = res["results"]

        assert "source-1" in results and results["source-1"]["status"] == "success"
        assert "filter-1" in results and results["filter-1"]["status"] == "success"
        assert "weibull-1" in results and results["weibull-1"]["status"] == "success"

        weibull_output = results["weibull-1"]["output"]
        assert "beta" in weibull_output and weibull_output["beta"] is not None
        assert "eta" in weibull_output and weibull_output["eta"] is not None

    def test_template_kijima_repair(self, client: TestClient):
        """Test Kijima Repair Template: DataSource -> Filter -> Kijima Multi-model"""
        payload = {
            "nodes": [
                {"id": "source-2", "type": "dataSource", "data": {}, "x": 50, "y": 150},
                {
                    "id": "filter-2",
                    "type": "filter",
                    "data": {"equipment": "Motor A"},
                    "x": 280,
                    "y": 150,
                },
                {
                    "id": "kijima-2",
                    "type": "kijima",
                    "data": {"model_types": [1, 2, 3]},
                    "x": 520,
                    "y": 150,
                },
            ],
            "edges": [
                {"id": "edge-3", "source": "source-2", "target": "filter-2"},
                {"id": "edge-4", "source": "filter-2", "target": "kijima-2"},
            ],
        }

        response = client.post("/api/workbench/execute", json=payload)
        assert response.status_code == 200
        res = response.json()
        assert res["status"] == "success"
        results = res["results"]

        assert "kijima-2" in results and results["kijima-2"]["status"] == "success"
        kijima_output = results["kijima-2"]["output"]
        assert "models" in kijima_output
        assert len(kijima_output["models"]) == 3
        for m in kijima_output["models"]:
            assert "beta" in m and "eta" in m and "model_name" in m

    def test_template_ram_simulation(self, client: TestClient):
        """Test RAM Simulation Template: DataSource -> Filter -> Weibull -> RAM Simulator"""
        payload = {
            "nodes": [
                {"id": "source-3", "type": "dataSource", "data": {}, "x": 50, "y": 150},
                {
                    "id": "filter-3",
                    "type": "filter",
                    "data": {"equipment": "Motor A"},
                    "x": 250,
                    "y": 150,
                },
                {"id": "weibull-3", "type": "weibull", "data": {}, "x": 450, "y": 50},
                {"id": "ram-3", "type": "ram", "data": {}, "x": 650, "y": 150},
            ],
            "edges": [
                {"id": "edge-5", "source": "source-3", "target": "filter-3"},
                {"id": "edge-6", "source": "filter-3", "target": "weibull-3"},
                {"id": "edge-7", "source": "weibull-3", "target": "ram-3"},
            ],
        }

        response = client.post("/api/workbench/execute", json=payload)
        assert response.status_code == 200
        res = response.json()
        assert res["status"] == "success"
        results = res["results"]

        assert "ram-3" in results and results["ram-3"]["status"] == "success"
        ram_output = results["ram-3"]["output"]
        assert "availability" in ram_output
        assert ram_output["availability"] > 0

    def test_template_pareto_analysis(self, client: TestClient):
        """Test Pareto & Jackknife Template: DataSource -> Filter -> Pareto & Jackknife"""
        payload = {
            "nodes": [
                {"id": "source-4", "type": "dataSource", "data": {}, "x": 50, "y": 150},
                {
                    "id": "filter-4",
                    "type": "filter",
                    "data": {"equipment": ""},
                    "x": 250,
                    "y": 150,
                },
                {
                    "id": "pareto-4",
                    "type": "pareto",
                    "data": {"group_by": "Equipment"},
                    "x": 450,
                    "y": 50,
                },
                {
                    "id": "jackknife-4",
                    "type": "jackknife",
                    "data": {"compare_by": "Equipment"},
                    "x": 450,
                    "y": 250,
                },
            ],
            "edges": [
                {"id": "edge-8", "source": "source-4", "target": "filter-4"},
                {"id": "edge-9", "source": "filter-4", "target": "pareto-4"},
                {"id": "edge-10", "source": "filter-4", "target": "jackknife-4"},
            ],
        }

        response = client.post("/api/workbench/execute", json=payload)
        assert response.status_code == 200
        res = response.json()
        assert res["status"] == "success"
        results = res["results"]

        assert "pareto-4" in results and results["pareto-4"]["status"] == "success"
        assert (
            "jackknife-4" in results and results["jackknife-4"]["status"] == "success"
        )

    def test_template_trend_flow(self, client: TestClient):
        """Test KPI Trend & Weibull Template: DataSource -> Filter -> KPI Trend & Weibull"""
        payload = {
            "nodes": [
                {"id": "source-5", "type": "dataSource", "data": {}, "x": 50, "y": 150},
                {
                    "id": "filter-5",
                    "type": "filter",
                    "data": {"equipment": "Motor A"},
                    "x": 250,
                    "y": 150,
                },
                {"id": "trend-5", "type": "trend", "data": {}, "x": 450, "y": 50},
                {"id": "weibull-5", "type": "weibull", "data": {}, "x": 450, "y": 250},
            ],
            "edges": [
                {"id": "edge-11", "source": "source-5", "target": "filter-5"},
                {"id": "edge-12", "source": "filter-5", "target": "trend-5"},
                {"id": "edge-13", "source": "filter-5", "target": "weibull-5"},
            ],
        }

        response = client.post("/api/workbench/execute", json=payload)
        assert response.status_code == 200
        res = response.json()
        assert res["status"] == "success"
        results = res["results"]

        assert "trend-5" in results and results["trend-5"]["status"] == "success"
        assert "weibull-5" in results and results["weibull-5"]["status"] == "success"
