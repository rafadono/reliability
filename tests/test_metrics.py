import math
import numpy as np
from unittest.mock import patch

from src.reliability_analysis.analysis.metrics import (
    kolmogorov_smirnov_test,
    mean_residual_life,
    r2_mrl,
    ks_test_weibull_pit,
    ks_test_kijima_pit,
)


@patch("src.reliability_analysis.analysis.metrics.kstest")
def test_kolmogorov_smirnov_test(mock_kstest):
    x = [1, 2, 3]
    beta = 2.0
    eta = 1.5
    mock_kstest.return_value = "kstest_result"

    result = kolmogorov_smirnov_test(x, beta, eta)

    mock_kstest.assert_called_once_with(x, "weibull_min", args=(beta, 0, eta))
    assert result == "kstest_result"


@patch("src.reliability_analysis.analysis.metrics.quad")
@patch("src.reliability_analysis.analysis.kijima_model.pdf")
def test_mean_residual_life(mock_pdf, mock_quad):
    v_prev = 1.0
    beta = 2.0
    eta = 1.5

    mock_quad.return_value = (42.0, 0.1)

    result = mean_residual_life(v_prev, beta, eta)

    mock_quad.assert_called_once()
    integrand = mock_quad.call_args[0][0]

    mock_pdf.return_value = 5.0
    val = integrand(2.0)

    mock_pdf.assert_called_once_with(2.0, v_prev, beta, eta)
    assert val == 2.0 * 5.0
    assert result == 42.0


@patch("src.reliability_analysis.analysis.metrics.mean_residual_life")
@patch("src.reliability_analysis.analysis.metrics.calculate_virtual_age")
def test_r2_mrl(mock_calculate_virtual_age, mock_mrl):
    x = np.array([10.0, 20.0, 30.0])
    delta = np.array([1, 0, 1])
    ar = 0.5
    ap = 0.5
    beta = 2.0
    eta = 1.5
    model_type = 1

    mock_calculate_virtual_age.return_value = np.array([5.0, 10.0, 15.0])
    mock_mrl.side_effect = [100.0, 200.0, 300.0]

    result = r2_mrl(x, delta, ar, ap, beta, eta, model_type)

    mock_calculate_virtual_age.assert_called_once_with(x, delta, ar, ap, model_type)

    # mean_residual_life is called for the V_prev elements: [0.0, 5.0, 10.0]
    assert mock_mrl.call_count == 3
    mock_mrl.assert_any_call(0.0, beta, eta)
    mock_mrl.assert_any_call(5.0, beta, eta)
    mock_mrl.assert_any_call(10.0, beta, eta)

    y_pred = np.array([100.0, 200.0, 300.0])
    y_obs = x
    mean_obs = np.mean(y_obs)
    ss_tot = np.sum((y_obs - mean_obs) ** 2)
    ss_res = np.sum((y_obs - y_pred) ** 2)
    expected_r2 = 1 - ss_res / ss_tot

    assert math.isclose(result, expected_r2)


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


@patch("src.reliability_analysis.analysis.metrics.kstest")
@patch("src.reliability_analysis.analysis.metrics.calculate_virtual_age")
def test_ks_test_kijima_pit(mock_calculate_virtual_age, mock_kstest):
    x = np.array([30.0, 10.0, 20.0])
    delta = np.array([1, 1, 0])
    beta = 2.0
    eta = 1.5
    ar = 0.5
    ap = 0.5
    model_type = 1

    mock_kstest.return_value = "kijima_pit_result"
    mock_calculate_virtual_age.return_value = np.array([5.0, 10.0, 15.0])

    result = ks_test_kijima_pit(x, delta, beta, eta, ar, ap, model_type)

    sorted_x = np.sort(x)
    # V is [5.0, 10.0, 15.0]
    # S = np.exp((V[-1]/eta)**beta - ((V[-1] + x)/eta)**beta)
    # V[-1] is 15.0
    expected_S = np.exp((15.0 / eta) ** beta - ((15.0 + sorted_x) / eta) ** beta)
    expected_F = 1 - expected_S
    # delta after sort (x was [30.0, 10.0, 20.0], sorted is [10.0, 20.0, 30.0])
    # sorted_x is [10.0, 20.0, 30.0]. Original delta index matching sorted_x:
    # 10.0 was index 1 -> delta=1
    # 20.0 was index 2 -> delta=0
    # 30.0 was index 0 -> delta=1
    # So sorted_delta is [1, 0, 1]
    sorted_delta = np.array([1, 0, 1])
    expected_F_fail = expected_F[sorted_delta == 1]

    mock_kstest.assert_called_once()
    call_args = mock_kstest.call_args[0]
    np.testing.assert_array_almost_equal(call_args[0], expected_F_fail)
    assert call_args[1] == "uniform"
    assert result == "kijima_pit_result"
