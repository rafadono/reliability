import numpy as np
from numba import njit, prange
from scipy.signal import lfilter


#precalcular pesos de Kijima 1
def _compute_weights(ar, ap, delta):
    # Precalcula ar**delta * ap**(1-delta)
    return np.asarray(ar, float)**delta * np.asarray(ap, float)**(1.0 - delta)

# Kijima 1
def calculate_ki(x, w):
    """
    KI vectorizado: V = cumsum(w * x)
    """
    return np.cumsum(w * x)

# Kijima 2
@njit(parallel=True)
def _calculate_k2_general(x, ar, ap):
    n = x.size
    V = np.zeros(n, dtype=x.dtype)
    for i in prange(n):
        s = 0.0
        for j in range(i + 1):
            expo = i - j + 1
            s += x[j] * (ar[j]**expo) * (ap[j]**expo)
        V[i] = s
    return V

def calculate_k2(x, ar, ap):
    """KII: O(n) if scalars, else Numba-accelerated O(n^2)"""
    if np.ndim(ar) == 0 and np.ndim(ap) == 0:
        w = float(ar) * float(ap)
        return lfilter([w], [1.0, -w], x)
    return _calculate_k2_general(x, np.asarray(ar, float), np.asarray(ap, float))

# Edad virtual
def calculate_virtual_age(x, delta, ar, ap, model_type):
    """
    Wrapper que elige entre modelo KI (1) o KII (2).
    """
    x = np.asarray(x, float)
    delta = np.asarray(delta, float)
    model = int(model_type[0]) if isinstance(model_type, (list, tuple)) else int(model_type)
    if model == 1:
        w = _compute_weights(ar, ap, delta)
        return calculate_ki(x, w)
    elif model == 2:
        return calculate_k2(x, ar, ap)
    else:
        raise ValueError(f"Invalid model_type: {model}")

# PDF y Confiabilidad
def pdf(t, v, beta, eta):
    t = np.asarray(t, float)
    return (beta/eta) * ((v + t)/eta)**(beta - 1) * np.exp((v/eta)**beta - ((v + t)/eta)**beta)

def reliability(t, v, beta, eta):
    t = np.asarray(t, float)
    return np.exp((v/eta)**beta - ((v + t)/eta)**beta)

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