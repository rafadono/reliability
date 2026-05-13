import numpy as np
from scipy.stats import kstest
from app.kijima_model import calculate_virtual_age

# Métricas: AIC, BIC, R², KS-test
def calculate_aic_bic(log_like, k, n):
    aic = 2*k - 2*log_like
    bic = np.log(n)*k - 2*log_like
    return aic, bic

def kolmogorov_smirnov_test(x, beta, eta):
    return kstest(x, 'weibull_min', args=(beta, 0, eta))

def ks_test_weibull_pit(x, beta, eta):
    # PIT para Weibull puro
    F = 1 - np.exp(-(x/eta)**beta)
    return kstest(F, 'uniform')

def ks_test_kijima_pit(x, delta, beta, eta, ar, ap, model_type):
    x = np.sort(x)
    V = calculate_virtual_age(x, delta, ar, ap, model_type)
    # PIT condicionado
    S = np.exp((V[-1]/eta)**beta - ((V[-1] + x)/eta)**beta)
    F = 1 - S
    # Sólo comparas las FALLAS
    F_fail = F[delta == 1]
    return kstest(F_fail, 'uniform')
