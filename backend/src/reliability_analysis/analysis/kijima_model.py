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


@njit(cache=True)
def calculate_ki_td(
    x: np.ndarray, delta: np.ndarray, ar: float, ap: float, br: float, bp: float
) -> np.ndarray:
    """Kijima I TD: V_i = V_{i-1} + q_i(T_i) * x_i"""
    V = np.zeros_like(x)
    v_prev = 0.0
    cum_t = 0.0
    for i in range(len(x)):
        cum_t += x[i]
        q0 = ar if delta[i] else ap
        b = br if delta[i] else bp
        q = q0 + b * cum_t
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0
        v_i = v_prev + q * x[i]
        V[i] = v_i
        v_prev = v_i
    return V


@njit(cache=True)
def calculate_k2_td(
    x: np.ndarray, delta: np.ndarray, ar: float, ap: float, br: float, bp: float
) -> np.ndarray:
    """Kijima II TD: V_i = q_i(T_i) * (V_{i-1} + x_i)"""
    V = np.zeros_like(x)
    v_prev = 0.0
    cum_t = 0.0
    for i in range(len(x)):
        cum_t += x[i]
        q0 = ar if delta[i] else ap
        b = br if delta[i] else bp
        q = q0 + b * cum_t
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0
        v_i = q * (v_prev + x[i])
        V[i] = v_i
        v_prev = v_i
    return V


def calculate_virtual_age(
    x: np.ndarray,
    delta: np.ndarray,
    ar: float,
    ap: float,
    model_type: int,
    br: float = 0.0,
    bp: float = 0.0,
) -> np.ndarray:
    """Calculate virtual age for Kijima I or II model (constant or time-dependent)."""
    x = np.asarray(x, float)
    delta = np.asarray(delta, float)
    model = (
        int(model_type[0]) if isinstance(model_type, (list, tuple)) else int(model_type)
    )

    if model == 1:
        return calculate_ki(x, delta, ar, ap)
    elif model == 2:
        return calculate_k2(x, delta, ar, ap)
    elif model == 3:
        return calculate_ki_td(x, delta, ar, ap, br, bp)
    elif model == 4:
        return calculate_k2_td(x, delta, ar, ap, br, bp)
    else:
        raise ValueError(f"Invalid model_type: {model}")


def reliability(t: np.ndarray, V: float, beta: float, eta: float) -> np.ndarray:
    """Weibull reliability function with virtual age."""
    t = np.asarray(t, dtype=float)
    return np.exp((V / eta) ** beta - ((V + t) / eta) ** beta)


def pdf(t: np.ndarray, V: float, beta: float, eta: float) -> np.ndarray:
    """Probability density function."""
    t = np.asarray(t, dtype=float)
    base = (V + t) / eta
    return (beta / eta) * base ** (beta - 1) * np.exp((V / eta) ** beta - base**beta)


def hazard(t, V, beta, eta):
    """Failure rate (hazard rate)."""
    return (beta / eta) * ((V + t) / eta) ** (beta - 1)


@njit(parallel=True, cache=True)
def _neg_loglik(x, delta, beta, eta, ar, ap, model_type):
    """Negative log-likelihood for optimization."""
    n = x.size
    V_prev = 0.0
    neg_ll = 0.0
    inv_eta = 1.0 / eta
    log_beta = np.log(beta)
    log_eta = np.log(eta)

    neg_ll += -delta.sum() * log_beta
    neg_ll += delta.sum() * beta * log_eta

    if model_type == 2:
        w = ar * ap

    for i in range(n):
        di = delta[i]
        xi = x[i]

        if model_type == 1:
            wi = (ar**di) * (ap ** (1 - di))
            V_i = V_prev + wi * xi
        else:
            V_i = w * (xi + V_prev)

        neg_ll -= di * (beta - 1) * np.log(V_prev + xi)
        neg_ll -= (V_prev * inv_eta) ** beta - ((V_prev + xi) * inv_eta) ** beta

        V_prev = V_i

    return neg_ll


@njit(parallel=True, cache=True)
def _neg_loglik_td(x, delta, beta, eta, ar, ap, br, bp, model_type):
    """Negative log-likelihood for optimization of time-dependent Kijima models."""
    n = x.size
    V_prev = 0.0
    neg_ll = 0.0
    inv_eta = 1.0 / eta
    log_beta = np.log(beta)
    log_eta = np.log(eta)

    neg_ll += -delta.sum() * log_beta
    neg_ll += delta.sum() * beta * log_eta

    cum_t = 0.0
    for i in range(n):
        di = delta[i]
        xi = x[i]
        cum_t += xi

        # Corrective (di=1) or Preventive (di=0)
        q0 = ar if di else ap
        b = br if di else bp
        q = q0 + b * cum_t
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0

        if model_type == 3:  # Kijima I TD
            V_i = V_prev + q * xi
        else:                # Kijima II TD
            V_i = q * (xi + V_prev)

        neg_ll -= di * (beta - 1) * np.log(V_prev + xi)
        neg_ll -= (V_prev * inv_eta) ** beta - ((V_prev + xi) * inv_eta) ** beta

        V_prev = V_i

    return neg_ll


def virtual_age_ratio(
    x: np.ndarray,
    delta: np.ndarray,
    ar: float,
    ap: float,
    model_type: int,
    br: float = 0.0,
    bp: float = 0.0,
) -> float:
    """
    Calculate average V_i / T_i for a given Kijima model.
    - x: TBX array
    - delta: 1 for corrective actions, 0 for preventive
    - ar, ap: Kijima parameters
    - model_type: 1, 2, 3, or 4
    """
    V = calculate_virtual_age(x, delta, ar, ap, model_type, br, bp)
    T = np.cumsum(x)
    return np.mean(V / T)


def auc_improvement(beta: float, eta: float, Vn: float, t_max: float = None) -> float:
    """
    Compare Kijima reliability AUC vs pure Weibull (ap=ar=1).
    - beta, eta: Weibull parameters
    - Vn: virtual age at the end of the period
    - t_max: evaluation limit (defaults to cumulative TBX)
    """
    if t_max is None:
        raise ValueError("Define t_max para la evaluación del AUC")
    t = np.linspace(0, t_max, 200)
    R_k = reliability(t, Vn, beta, eta)
    R_w = np.exp(-((t / eta) ** beta))
    auc_k = np.trapz(R_k, t)
    auc_w = np.trapz(R_w, t)
    return (auc_k - auc_w) / auc_w
