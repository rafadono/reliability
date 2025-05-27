import numpy as np
from scipy.stats import kstest
from scipy.integrate import quad
from reliability.Nonparametric import KaplanMeier
from app.kijima_model import pdf, calculate_virtual_age

# Métricas: AIC, BIC, R², KS-test
def calculate_aic_bic(log_like, k, n):
    aic = 2*k - 2*log_like
    bic = np.log(n)*k - 2*log_like
    return aic, bic

def calculate_r2(x, V):
    ss_tot = np.sum((x - x.mean()) ** 2)
    ss_res = np.sum((x - V) ** 2)
    return 1.0 - ss_res / ss_tot

def kolmogorov_smirnov_test(x, beta, eta):
    return kstest(x, 'weibull_min', args=(beta, 0, eta))

def calculate_r2_kijima_km(x, delta, V, beta, eta):
    # 1) tiempos efectivos
    n = x.size
    T_eff = np.empty(n, float)
    for i in range(n):
        v_prev   = V[i-1] if i > 0 else 0.0
        T_eff[i] = v_prev + x[i]

    # 2) separa fallas/censuras
    failures       = T_eff[delta == 1]
    right_censored = T_eff[delta == 0]

    # 3) ajusta Kaplan–Meier
    km = KaplanMeier(
        failures=failures,
        right_censored=right_censored,
        show_plot=False,
        print_results=False
    )

    # 4) usa la función stepwise SF (SF) y sus xvals
    times = km.xvals
    S_km  = km.SF

    # 5) Calculo de la curva de confiabilidad
    Vn      = float(V[-1])
    R_model = np.exp((Vn/eta)**beta - ((Vn + times)/eta)**beta)

    # 6) R² clásico
    ss_tot = np.sum((S_km - S_km.mean())**2)
    ss_res = np.sum((S_km - R_model)**2)
    return 1.0 - ss_res/ss_tot

def mean_residual_life(v_prev, beta, eta):
    def integrand(t: float) -> float:
        return t * pdf(t, v_prev, beta, eta)

    mrl, _ = quad(integrand, 0, np.inf)
    return mrl

def r2_mrl(x, delta, ar, ap, beta, eta, model_type):
    V = calculate_virtual_age(x, delta, ar, ap, model_type)
    n = x.size
    y_pred = np.empty(n)
    for i in range(n):
        v_prev = V[i-1] if i>0 else 0.0
        y_pred[i] = mean_residual_life(v_prev, beta, eta)
    # Mismo y_obs para todos:
    y_obs = x
    ss_tot = np.sum((y_obs - y_obs.mean())**2)
    ss_res = np.sum((y_obs - y_pred)**2)
    return 1 - ss_res/ss_tot

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
