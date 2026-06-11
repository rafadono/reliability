"""
Unit tests for ReliabilityFitter and KijimaFitter.
"""

import pytest
import numpy as np
import pandas as pd
from src.reliability_analysis.analysis.models import ReliabilityFitter, KijimaFitter


@pytest.fixture
def sample_fit_data():
    """Create sample data for fitting."""
    np.random.seed(42)
    tbx_data = np.random.weibull(2, 30) * 100  # Weibull data
    ttx_data = np.random.weibull(1.5, 30) * 50

    data = {
        "TBX": tbx_data,
        "TTX": ttx_data,
        "mdf": ["Failure_A"] * 20 + ["Failure_B"] * 10,
    }
    return pd.DataFrame(data)


class TestReliabilityFitter:
    def test_fitter_initialization(self):
        """Verify ReliabilityFitter initializes correctly."""
        fitter = ReliabilityFitter()
        assert fitter.excluded_models is not None
        assert len(fitter.excluded_models) > 0

    def test_fit_returns_dict(self, sample_fit_data):
        """Verify fit returns a dictionary."""
        fitter = ReliabilityFitter()
        result = fitter.fit(sample_fit_data, "TBX", ["Failure_B"])

        assert isinstance(result, dict)

    def test_fit_required_keys(self, sample_fit_data):
        """Verify fit returns all required keys."""
        fitter = ReliabilityFitter()
        result = fitter.fit(sample_fit_data, "TBX", ["Failure_B"])

        required_keys = [
            "best_distribution",
            "name",
            "parameters",
            "mean",
            "std_dev",
            "AICc",
            "BIC",
        ]
        assert all(key in result for key in required_keys)

    def test_fit_distribution_name_not_empty(self, sample_fit_data):
        """Verify a distribution name is returned."""
        fitter = ReliabilityFitter()
        result = fitter.fit(sample_fit_data, "TBX", ["Failure_B"])

        assert isinstance(result["name"], str)
        assert len(result["name"]) > 0

    def test_fit_mean_is_positive(self, sample_fit_data):
        """Verify mean is positive."""
        fitter = ReliabilityFitter()
        result = fitter.fit(sample_fit_data, "TBX", ["Failure_B"])

        assert result["mean"] > 0

    def test_fit_std_dev_is_positive(self, sample_fit_data):
        """Verify standard deviation is positive."""
        fitter = ReliabilityFitter()
        result = fitter.fit(sample_fit_data, "TBX", ["Failure_B"])

        assert result["std_dev"] > 0

    def test_fit_metrics_are_numbers(self, sample_fit_data):
        """Verify AIC and BIC are numbers."""
        fitter = ReliabilityFitter()
        result = fitter.fit(sample_fit_data, "TBX", ["Failure_B"])

        assert isinstance(result["AICc"], (int, float)) and not np.isnan(result["AICc"])
        assert isinstance(result["BIC"], (int, float)) and not np.isnan(result["BIC"])

    def test_fit_with_all_censored(self, sample_fit_data):
        """Verify behavior when all data is censored."""
        fitter = ReliabilityFitter()

        # If we try to censor everything, it should fall back to using uncensored only
        result = fitter.fit(sample_fit_data, "TBX", ["Failure_A"])

        assert result["name"] is not None

    def test_fit_ttx_column(self, sample_fit_data):
        """Verify fit works with TTX column."""
        fitter = ReliabilityFitter()
        result = fitter.fit(sample_fit_data, "TTX", ["Failure_B"])

        assert result["mean"] > 0
        assert result["std_dev"] > 0


class TestKijimaFitter:
    def test_kijima_initialization(self):
        """Verify KijimaFitter initializes."""
        kijima = KijimaFitter()
        assert kijima.models is not None

    def test_fit_single_model_returns_dict(self, sample_fit_data):
        """Verify fit with single model returns dict."""
        kijima = KijimaFitter()
        result = kijima.fit(sample_fit_data, "TBX", ["Failure_B"], models=1)

        assert isinstance(result, dict)

    def test_fit_multiple_models_returns_list(self, sample_fit_data):
        """Verify fit with multiple models returns list."""
        kijima = KijimaFitter()
        result = kijima.fit(sample_fit_data, "TBX", ["Failure_B"], models=[1, 2])

        assert isinstance(result, list)
        assert len(result) == 2

    def test_fit_required_keys_single_model(self, sample_fit_data):
        """Verify keys in single model result."""
        kijima = KijimaFitter()
        result = kijima.fit(sample_fit_data, "TBX", ["Failure_B"], models=1)

        required_keys = [
            "model_name",
            "beta",
            "eta",
            "ar",
            "ap",
            "AIC",
            "BIC",
            "mean",
            "std",
            "t",
            "R",
            "failure_rate",
        ]
        assert all(key in result for key in required_keys)

    def test_fit_beta_eta_positive(self, sample_fit_data):
        """Verify beta and eta are positive."""
        kijima = KijimaFitter()
        result = kijima.fit(sample_fit_data, "TBX", ["Failure_B"], models=1)

        assert result["beta"] > 0
        assert result["eta"] > 0

    def test_fit_ar_ap_in_range(self, sample_fit_data):
        """Verify ar and ap are in range [0, 1]."""
        kijima = KijimaFitter()
        result = kijima.fit(sample_fit_data, "TBX", ["Failure_B"], models=1)

        assert 0 <= result["ar"] <= 1
        assert 0 <= result["ap"] <= 1

    def test_fit_mean_positive(self, sample_fit_data):
        """Verify mean is positive."""
        kijima = KijimaFitter()
        result = kijima.fit(sample_fit_data, "TBX", ["Failure_B"], models=1)

        assert result["mean"] > 0

    def test_fit_curves_same_length(self, sample_fit_data):
        """Verify t, R, and failure_rate have the same length."""
        kijima = KijimaFitter()
        result = kijima.fit(sample_fit_data, "TBX", ["Failure_B"], models=1)

        assert len(result["t"]) == len(result["R"]) == len(result["failure_rate"])

    def test_fit_reliability_in_range(self, sample_fit_data):
        """Verify reliability is in range [0, 1]."""
        kijima = KijimaFitter()
        result = kijima.fit(sample_fit_data, "TBX", ["Failure_B"], models=1)

        assert np.all((result["R"] >= 0) & (result["R"] <= 1))

    def test_fit_failure_rate_positive(self, sample_fit_data):
        """Verify failure rate is positive."""
        kijima = KijimaFitter()
        result = kijima.fit(sample_fit_data, "TBX", ["Failure_B"], models=1)

        assert np.all(result["failure_rate"] >= 0)

    def test_model_names_correct(self, sample_fit_data):
        """Verify model names are correct."""
        kijima = KijimaFitter()
        result = kijima.fit(sample_fit_data, "TBX", ["Failure_B"], models=[1, 2])

        assert result[0]["model_name"] == "Kijima I"
        assert result[1]["model_name"] == "Kijima II"
