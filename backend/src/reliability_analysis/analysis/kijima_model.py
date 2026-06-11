import numpy as np
from numba import njit

@njit(cache=True)
def calculate_ki(x: np.ndarray, delta: np.ndarray, ar: float, ap: float) -> np.ndarray:
    """Kijima I: V_i = V_{i-1} + a_i * x_i"""
    a = np.where(delta == 1.0, ar, ap)
    return np.cumsum(a * x)

@njit(cache=True)
def calculate_k2(x: np.ndarray, delta: np.ndarray, ar: float, ap: float) -> np.ndarray:
    """Kijima II: V_i = a * (V_{i-1} + x_i)"""
    V = np.zeros_like(x)
    v_prev = 0.0
    for i in range(len(x)):
        a = ar if delta[i] else ap
        v_i = a * (v_prev + x[i])
        V[i] = v_i
        v_prev = v_i
    return V

def calculate_virtual_age(x: np.ndarray, delta: np.ndarray, ar: float, ap: float, model_type: int) -> np.ndarray:
    """Calcula edad virtual según modelo Kijima I o II"""
    x = np.asarray(x, float)
    delta = np.asarray(delta, float)
    model = int(model_type[0]) if isinstance(model_type, (list, tuple)) else int(model_type)

    if model == 1:
        return calculate_ki(x, delta, ar, ap)
    elif model == 2:
        return calculate_k2(x, delta, ar, ap)
    else:
        raise ValueError(f"Invalid model_type: {model}")

def reliability(t: np.ndarray, V: float, beta: float, eta: float) -> np.ndarray:
    """Función de confiabilidad Weibull con edad virtual"""
    t = np.asarray(t, dtype=float)
    return np.exp((V / eta)**beta - ((V + t) / eta)**beta)

def pdf(t: np.ndarray, V: float, beta: float, eta: float) -> np.ndarray:
    """Función de densidad de probabilidad"""
    t = np.asarray(t, dtype=float)
    base = (V + t) / eta
    return (beta / eta) * base**(beta - 1) * np.exp((V / eta)**beta - base**beta)

def hazard(t, V, beta, eta):
    """Tasa de falla (hazard rate)"""
    return (beta / eta) * ((V + t) / eta) ** (beta - 1)

@njit(parallel=True, cache=True)
def _neg_loglik(x, delta, beta, eta, ar, ap, model_type):
    """Log-verosimilitud negativa para optimización"""
    n = x.size
    V_prev = 0.0
    neg_ll = 0.0
    inv_eta = 1.0 / eta
    log_beta = np.log(beta)
    log_eta  = np.log(eta)

    neg_ll += - delta.sum() * log_beta
    neg_ll +=   delta.sum() * beta * log_eta

    if model_type == 2:
        w = ar * ap

    for i in range(n):
        di = delta[i]
        xi = x[i]

        if model_type == 1:
            wi = (ar**di) * (ap**(1 - di))
            V_i = V_prev + wi * xi
        else:
            V_i = w * (xi + V_prev)

        neg_ll -= di * (beta - 1) * np.log(V_prev + xi)
        neg_ll -= ((V_prev * inv_eta)**beta - ((V_prev + xi) * inv_eta)**beta)

        V_prev = V_i

    return neg_ll

def virtual_age_ratio(x: np.ndarray, delta: np.ndarray, ar: float, ap: float, model_type: int) -> float:
    """
    Calcula el promedio de V_i / T_i para un modelo Kijima dado.
    - x: array de TBX
    - delta: 1 para correcciones, 0 para preventivas
    - ar, ap: parámetros de Kijima
    - model_type: 1 o 2
    """
    V = calculate_virtual_age(x, delta, ar, ap, model_type)
    T = np.cumsum(x)
    return np.mean(V / T)

def auc_improvement(beta: float, eta: float, Vn: float, t_max: float = None) -> float:
    """
    Compara el AUC de confiabilidad Kijima vs Weibull puro (ap=ar=1).
    - beta, eta: parámetros Weibull
    - Vn: edad virtual al final del periodo
    - t_max: límite para el cálculo (por defecto hasta suma TBX)
    """
    if t_max is None:
        raise ValueError("Define t_max para la evaluación del AUC")
    t = np.linspace(0, t_max, 200)
    R_k = reliability(t, Vn, beta, eta)
    R_w = np.exp(- (t/eta)**beta)
    auc_k = np.trapz(R_k, t)
    auc_w = np.trapz(R_w, t)
    return (auc_k - auc_w) / auc_w