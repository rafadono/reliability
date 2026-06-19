import numpy as np
from scipy.stats import kstest


def calculate_aic_bic(log_like: float, k: int, n: int) -> tuple[float, float]:
    aic = 2 * k - 2 * log_like
    bic = np.log(n) * k - 2 * log_like
    return aic, bic


def ks_test_weibull_pit(x: np.ndarray, beta: float, eta: float) -> tuple[float, float]:
    """PIT-based KS test for a pure Weibull fit."""
    F = 1 - np.exp(-((x / eta) ** beta))
    return kstest(F, "uniform")
