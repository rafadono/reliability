import numpy as np
from scipy.stats import kstest
from scipy.integrate import quad
from src.reliability_analysis.analysis.kijima_model import calculate_virtual_age


# Metrics: AIC, BIC, R2, KS-test
def calculate_aic_bic(log_like, k, n):
    aic = 2 * k - 2 * log_like
    bic = np.log(n) * k - 2 * log_like
    return aic, bic


def kolmogorov_smirnov_test(x, beta, eta):
    return kstest(x, "weibull_min", args=(beta, 0, eta))


def ks_test_weibull_pit(x, beta, eta):
    # PIT for pure Weibull
    F = 1 - np.exp(-((x / eta) ** beta))
    return kstest(F, "uniform")


def ks_test_kijima_pit(x, delta, beta, eta, ar, ap, model_type, br=0.0, bp=0.0):
    x = np.sort(x)
    V = calculate_virtual_age(x, delta, ar, ap, model_type, br, bp)
    # Conditional PIT
    S = np.exp((V[-1] / eta) ** beta - ((V[-1] + x) / eta) ** beta)
    F = 1 - S
    # Compare failures only
    F_fail = F[delta == 1]
    return kstest(F_fail, "uniform")


def mean_residual_life(v_prev, beta, eta):
    from src.reliability_analysis.analysis.kijima_model import pdf

    res, _ = quad(lambda t: t * pdf(t, v_prev, beta, eta), 0, np.inf)
    return res


def r2_mrl(x, delta, ar, ap, beta, eta, model_type):
    V = calculate_virtual_age(x, delta, ar, ap, model_type)
    V_prev = np.insert(V[:-1], 0, 0.0)

    y_pred = np.array([mean_residual_life(v, beta, eta) for v in V_prev])
    y_obs = np.asarray(x)

    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    ss_res = np.sum((y_obs - y_pred) ** 2)

    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
