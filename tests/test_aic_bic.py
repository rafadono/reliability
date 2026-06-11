import math
from src.reliability_analysis.analysis.metrics import calculate_aic_bic

def test_calculate_aic_bic_standard():
    log_like = -10
    k = 2
    n = 100
    aic, bic = calculate_aic_bic(log_like, k, n)
    assert aic == 2*k - 2*log_like
    assert bic == math.log(n)*k - 2*log_like

def test_calculate_aic_bic_n_one():
    log_like = -10
    k = 2
    n = 1
    aic, bic = calculate_aic_bic(log_like, k, n)
    # 2*2 - 2*(-10) = 4 + 20 = 24
    # log(1)*2 - 2*(-10) = 0*2 + 20 = 20
    assert aic == 24
    assert bic == 20

def test_calculate_aic_bic_log_like_zero():
    log_like = 0
    k = 2
    n = 100
    aic, bic = calculate_aic_bic(log_like, k, n)
    # 2*2 - 0 = 4
    # log(100)*2 - 0 = 4.605*2 = 9.210
    assert aic == 4
    assert math.isclose(bic, 2 * math.log(100))

def test_calculate_aic_bic_k_zero():
    log_like = -10
    k = 0
    n = 100
    aic, bic = calculate_aic_bic(log_like, k, n)
    # 2*0 - 2*(-10) = 20
    # log(100)*0 - 2*(-10) = 20
    assert aic == 20
    assert bic == 20

def test_calculate_aic_bic_large_values():
    log_like = -1e6
    k = 100
    n = 1000000
    aic, bic = calculate_aic_bic(log_like, k, n)
    assert aic == 2*k - 2*log_like
    assert bic == math.log(n)*k - 2*log_like
