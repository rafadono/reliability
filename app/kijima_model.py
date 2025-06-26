import numpy as np
from numba import njit

#precalcular pesos de Kijima 1
#def _compute_weights(ar: float, ap: float, delta: np.ndarray) -> np.ndarray:
#    """
#    Retorna un vector de pesos según tipo de evento:
#    ar si es falla (delta=1), ap si es preventiva (delta=0).
#    """
#    delta = np.asarray(delta, float)
#    return ar**delta * ap**(1 - delta)

# Kijima 1
def calculate_ki(x: np.ndarray, delta: np.ndarray, ar: float, ap: float) -> np.ndarray:
    V = np.zeros_like(x, dtype=float)
    v_prev = 0.0
    for i in range(len(x)):
        a = ar if delta[i] == 1.0 else ap
        v_i = v_prev + a*x[i]
        #if v_i < v_prev:
        #    print(f"k1 [ERROR] v_{i} < v_{i-1}: {v_i:.3f} < {v_prev:.3f} | a={a}, x={x[i]}")
        V[i] = v_i
        v_prev = v_i
    return V

# Kijima 2
#@njit(parallel=True)
#def _calculate_k2_general(x, ar, ap):
#    n = x.size
#    V = np.zeros(n, dtype=x.dtype)
#    for i in prange(n):
#        s = 0.0
#        for j in range(i + 1):
#            expo = i - j + 1
#            s += x[j] * (ar[j]**expo) * (ap[j]**expo)
#        V[i] = s
#    return V

def calculate_k2(x: np.ndarray, delta: np.ndarray, ar: float, ap: float) -> np.ndarray:
    V = np.zeros_like(x, dtype=float)
    v_prev = 0.0
    for i in range(len(x)):
        a = ar if delta[i] else ap
        v_i = a * (v_prev + x[i])
        #if v_i < v_prev:
        #    print(f"k2 [ERROR] v_{i} < v_{i-1}: {v_i:.3f} < {v_prev:.3f} | a={a}, x={x[i]}")
        V[i] = v_i
        v_prev = v_i
    return V

# Edad virtual
def calculate_virtual_age(x: np.ndarray, delta: np.ndarray, ar: float, ap: float, model_type: int) -> np.ndarray:
    x = np.asarray(x, float)
    delta = np.asarray(delta, float)
    model = int(model_type[0]) if isinstance(model_type, (list, tuple)) else int(model_type)

    if model == 1:
        #w = _compute_weights(ar, ap, delta)
        return calculate_ki(x, delta, ar, ap)
    elif model == 2:
        return calculate_k2(x, delta, ar, ap)
    else:
        raise ValueError(f"Invalid model_type: {model}")
    
# PDF y Confiabilidad
def reliability(t: np.ndarray, V: float, beta: float, eta: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.exp((V / eta)**beta - ((V + t) / eta)**beta)

def pdf(t: np.ndarray, V: float, beta: float, eta: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    base = (V + t) / eta
    return (beta / eta) * base**(beta - 1) * np.exp((V / eta)**beta - base**beta)

def hazard(t, V, beta, eta):
    return (beta / eta) * ((V + t) / eta) ** (beta - 1)

# Log-likelihood
@njit(parallel=True, cache=True)
def _neg_loglik(x, delta, beta, eta, ar, ap, model_type):
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
        else:  # model_type == 2
            V_i = w * (xi + V_prev)

        neg_ll -= di * (beta - 1) * np.log(V_prev + xi)
        neg_ll -= ((V_prev * inv_eta)**beta - ((V_prev + xi) * inv_eta)**beta)

        V_prev = V_i

    return neg_ll

def maintenance_efficiency(ar: float, ap: float) -> dict:
    """
    Devuelve índices de eficiencia de Correctivo y Preventivo:
      E_CM = 1 - ar
      E_PM = 1 - ap
    """
    return {
        "E_CM": 1 - ar,
        "E_PM": 1 - ap
    }

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
    # Kijima usando edad virtual Vn como shift
    R_k = reliability(t, Vn, beta, eta)
    R_w = np.exp(- (t/eta)**beta)
    auc_k = np.trapz(R_k, t)
    auc_w = np.trapz(R_w, t)
    return (auc_k - auc_w) / auc_w