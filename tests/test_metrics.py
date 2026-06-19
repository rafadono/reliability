import numpy as np
import pytest
from unittest.mock import patch
from src.reliability_analysis.analysis.metrics import (
    calculate_aic_bic,
    ks_test_weibull_pit,
)


def test_calculate_aic_bic():
    """Verify AIC and BIC formulas."""
    log_like = -50.0
    k = 4
    n = 100
    aic, bic = calculate_aic_bic(log_like, k, n)
    assert aic == pytest.approx(2 * k - 2 * log_like)
    assert bic == pytest.approx(np.log(n) * k - 2 * log_like)


@patch("src.reliability_analysis.analysis.metrics.kstest")
def test_ks_test_weibull_pit(mock_kstest):
    x = np.array([1.0, 2.0])
    beta = 2.0
    eta = 1.5
    mock_kstest.return_value = "weibull_pit_result"

    result = ks_test_weibull_pit(x, beta, eta)

    expected_F = 1 - np.exp(-((x / eta) ** beta))
    mock_kstest.assert_called_once()
    call_args = mock_kstest.call_args[0]
    np.testing.assert_array_almost_equal(call_args[0], expected_F)
    assert call_args[1] == "uniform"
    assert result == "weibull_pit_result"


def test_ks_test_kijima_pit_via_model():
    """KS-PIT is now KijimaModel.ks_test_pit() — verify end-to-end."""
    from src.reliability_analysis.analysis.kijima_model import KijimaModelI
    x = np.array([100.0, 150.0, 200.0, 80.0])
    delta = np.array([1.0, 1.0, 0.0, 1.0])
    m = KijimaModelI(beta=1.5, eta=500.0, ar=0.2, ap=0.8)
    stat, pval = m.ks_test_pit(x, delta)
    assert 0.0 <= stat <= 1.0
    assert 0.0 <= pval <= 1.0


def test_r2_mrl_via_model():
    """R2-MRL is now KijimaModel.r2_mrl() — verify result is finite and <= 1."""
    from src.reliability_analysis.analysis.kijima_model import KijimaModelI
    x = np.array([100.0, 150.0, 200.0])
    delta = np.array([1.0, 0.0, 1.0])
    m = KijimaModelI(beta=1.5, eta=500.0, ar=0.2, ap=0.8)
    r2 = m.r2_mrl(x, delta)
    assert np.isfinite(r2)
    assert r2 <= 1.0


def test_mean_residual_life_via_model():
    """mean_residual_life is now KijimaModel.mean_residual_life() — verify it agrees with analytical mean()."""
    from src.reliability_analysis.analysis.kijima_model import KijimaModelI
    m = KijimaModelI(beta=1.5, eta=500.0, ar=0.2, ap=0.8)
    V = 50.0
    num = m.mean_residual_life(V)
    ana = m.mean(V)
    assert num == pytest.approx(ana, rel=1e-4)
