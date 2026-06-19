import numpy as np
from numba import njit
from scipy.special import gamma, gammaincc
from scipy.stats import kstest


@njit(cache=True)
def _calculate_ki(x: np.ndarray, delta: np.ndarray, ar: float, ap: float) -> np.ndarray:
    """Kijima I kernel: V_i = V_{i-1} + a_i * x_i"""
    a = np.where(delta == 1.0, ar, ap)
    return np.cumsum(a * x)


@njit(cache=True)
def _calculate_k2(x: np.ndarray, delta: np.ndarray, ar: float, ap: float) -> np.ndarray:
    """Kijima II kernel: V_i = a * (V_{i-1} + x_i)"""
    V = np.zeros_like(x)
    v_prev = 0.0
    for i in range(len(x)):
        a = ar if delta[i] > 0.5 else ap
        v_i = a * (v_prev + x[i])
        V[i] = v_i
        v_prev = v_i
    return V


@njit(cache=True)
def _calculate_ki_td(
    x: np.ndarray, delta: np.ndarray, ar: float, ap: float, br: float, bp: float
) -> np.ndarray:
    """Kijima I TD kernel: V_i = V_{i-1} + q_i(T_i) * x_i"""
    V = np.zeros_like(x)
    v_prev = 0.0
    cum_t = 0.0
    for i in range(len(x)):
        cum_t += x[i]
        q0 = ar if delta[i] > 0.5 else ap
        b = br if delta[i] > 0.5 else bp
        q = 1.0 - (1.0 - q0) * np.exp(-b * cum_t)
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0
        v_i = v_prev + q * x[i]
        V[i] = v_i
        v_prev = v_i
    return V


@njit(cache=True)
def _calculate_k2_td(
    x: np.ndarray, delta: np.ndarray, ar: float, ap: float, br: float, bp: float
) -> np.ndarray:
    """Kijima II TD kernel: V_i = q_i(T_i) * (V_{i-1} + x_i)"""
    V = np.zeros_like(x)
    v_prev = 0.0
    cum_t = 0.0
    for i in range(len(x)):
        cum_t += x[i]
        q0 = ar if delta[i] > 0.5 else ap
        b = br if delta[i] > 0.5 else bp
        q = 1.0 - (1.0 - q0) * np.exp(-b * cum_t)
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0
        v_i = q * (v_prev + x[i])
        V[i] = v_i
        v_prev = v_i
    return V

@njit(cache=True)
def _calculate_ki_td2(
    x: np.ndarray, delta: np.ndarray, ar: float, ap: float, br: float, bp: float
) -> np.ndarray:
    """Kijima I TD2 kernel (Logistic): V_i = V_{i-1} + q_i(T_i) * x_i"""
    V = np.zeros_like(x)
    v_prev = 0.0
    cum_t = 0.0
    for i in range(len(x)):
        cum_t += x[i]
        q0 = ar if delta[i] > 0.5 else ap
        b = br if delta[i] > 0.5 else bp
        
        q0_clamped = max(1e-6, min(1.0 - 1e-6, q0))
        C = np.log(q0_clamped / (1.0 - q0_clamped))
        z = C + b * cum_t
        q = 1.0 / (1.0 + np.exp(-z))
        
        v_i = v_prev + q * x[i]
        V[i] = v_i
        v_prev = v_i
    return V

@njit(cache=True)
def _calculate_k2_td2(
    x: np.ndarray, delta: np.ndarray, ar: float, ap: float, br: float, bp: float
) -> np.ndarray:
    """Kijima II TD2 kernel (Logistic): V_i = q_i(T_i) * (V_{i-1} + x_i)"""
    V = np.zeros_like(x)
    v_prev = 0.0
    cum_t = 0.0
    for i in range(len(x)):
        cum_t += x[i]
        q0 = ar if delta[i] > 0.5 else ap
        b = br if delta[i] > 0.5 else bp
        
        q0_clamped = max(1e-6, min(1.0 - 1e-6, q0))
        C = np.log(q0_clamped / (1.0 - q0_clamped))
        z = C + b * cum_t
        q = 1.0 / (1.0 + np.exp(-z))
        
        v_i = q * (v_prev + x[i])
        V[i] = v_i
        v_prev = v_i
    return V



@njit(parallel=True, cache=True)
def _neg_loglik(x, delta, beta, eta, ar, ap, model_type):
    """Negative log-likelihood for optimization (models 1 and 2)."""
    n = x.size
    V_prev = 0.0
    neg_ll = 0.0
    inv_eta = 1.0 / eta
    log_beta = np.log(beta)
    log_eta = np.log(eta)

    neg_ll += -delta.sum() * log_beta
    neg_ll += delta.sum() * beta * log_eta

    for i in range(n):
        di = delta[i]
        xi = x[i]

        if model_type == 1:
            wi = (ar**di) * (ap ** (1 - di))
            V_i = V_prev + wi * xi
        else:
            wi = ar if di > 0.5 else ap
            V_i = wi * (xi + V_prev)

        if di > 0.0:
            val = V_prev + xi
            if val <= 0.0:
                val = 1e-10
            neg_ll -= di * (beta - 1) * np.log(val)
        neg_ll -= (V_prev * inv_eta) ** beta - ((V_prev + xi) * inv_eta) ** beta

        V_prev = V_i

    return neg_ll


@njit(parallel=True, cache=True)
def _neg_loglik_td(x, delta, beta, eta, ar, ap, br, bp, model_type):
    """Negative log-likelihood for optimization (time-dependent models 3, 4, 5, 6)."""
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

        q0 = ar if di > 0.5 else ap
        b = br if di > 0.5 else bp
        
        if model_type in (3, 4):
            q = 1.0 - (1.0 - q0) * np.exp(-b * cum_t)
        else:
            q0_clamped = max(1e-6, min(1.0 - 1e-6, q0))
            C = np.log(q0_clamped / (1.0 - q0_clamped))
            z = C + b * cum_t
            q = 1.0 / (1.0 + np.exp(-z))
            
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0

        if model_type in (3, 5):
            V_i = V_prev + q * xi
        else:
            V_i = q * (xi + V_prev)

        if di > 0.0:
            val = V_prev + xi
            if val <= 0.0:
                val = 1e-10
            neg_ll -= di * (beta - 1) * np.log(val)
        neg_ll -= (V_prev * inv_eta) ** beta - ((V_prev + xi) * inv_eta) ** beta

        V_prev = V_i

    return neg_ll


@njit(cache=True)
def _reconstruct_grp_curves(
    x: np.ndarray,
    delta: np.ndarray,
    V_full: np.ndarray,
    beta: float,
    eta: float,
    points_per_interval: int,
    future_time: float,
    future_points: int,
):
    n_intervals = x.size
    T = np.zeros(n_intervals + 1)
    for i in range(n_intervals):
        T[i + 1] = T[i] + x[i]

    total_pts = n_intervals * points_per_interval + future_points
    t_out = np.zeros(total_pts)
    R_out = np.zeros(total_pts)
    h_out = np.zeros(total_pts)
    f_out = np.zeros(total_pts)
    V_out = np.zeros(total_pts)

    idx = 0
    # 1. Historical cycles
    for i in range(n_intervals):
        V_start = V_full[i]
        x_i = x[i]
        T_start = T[i]

        for k in range(points_per_interval):
            f = k / (points_per_interval - 1)
            t_loc = f * x_i
            t_glob = T_start + t_loc
            v_t = V_start + t_loc

            r_val = np.exp((V_start / eta) ** beta - (v_t / eta) ** beta)
            v_clamped = v_t if v_t > 1e-6 else 1e-6
            h_val = (beta / eta) * (v_clamped / eta) ** (beta - 1)

            t_out[idx] = t_glob
            R_out[idx] = r_val
            h_out[idx] = h_val
            f_out[idx] = h_val * r_val
            V_out[idx] = v_t
            idx += 1

    # 2. Future cycle
    T_last = T[-1]
    V_last = V_full[-1]
    for k in range(future_points):
        f = k / (future_points - 1)
        t_loc = f * future_time
        t_glob = T_last + t_loc
        v_t = V_last + t_loc

        r_val = np.exp((V_last / eta) ** beta - (v_t / eta) ** beta)
        v_clamped = v_t if v_t > 1e-6 else 1e-6
        h_val = (beta / eta) * (v_clamped / eta) ** (beta - 1)

        t_out[idx] = t_glob
        R_out[idx] = r_val
        h_out[idx] = h_val
        f_out[idx] = h_val * r_val
        V_out[idx] = v_t
        idx += 1

    return t_out, R_out, h_out, f_out, V_out


def _safe_exp_gamma(a: float, x: float) -> float:
    """
    Computes exp(x) * Gamma(a, x) stably.
    For x <= 50.0 uses exact scipy; for x > 50.0 uses a 4th-order asymptotic expansion.
    """
    if x <= 50.0:
        return np.exp(x) * gammaincc(a, x) * gamma(a)
    term0 = 1.0
    term1 = (a - 1.0) / x
    term2 = term1 * (a - 2.0) / x
    term3 = term2 * (a - 3.0) / x
    term4 = term3 * (a - 4.0) / x
    return (x ** (a - 1.0)) * (term0 + term1 + term2 + term3 + term4)


class KijimaModel:
    """
    Base class for Kijima virtual age models.
    Provides Weibull-based survival, hazard, MTBF, and goodness-of-fit methods.
    Subclasses implement virtual_age() using the appropriate recurrence.
    """
    def __init__(self, beta: float, eta: float, ar: float, ap: float):
        self.beta = float(beta)
        self.eta = float(eta)
        self.ar = float(ar)
        self.ap = float(ap)
        self.model_type: int = 0
        self.model_name: str = "Kijima Base"

    def virtual_age(self, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sf(self, t: np.ndarray, V: float) -> np.ndarray:
        """Survival function: R(t | V) = P(T > t + V | T > V)"""
        t = np.asarray(t, dtype=float)
        return np.exp((V / self.eta) ** self.beta - ((V + t) / self.eta) ** self.beta)

    def reliability(self, t: np.ndarray, V: float) -> np.ndarray:
        """Alias for sf."""
        return self.sf(t, V)

    def pdf(self, t: np.ndarray, V: float) -> np.ndarray:
        """Probability density function f(t | V)"""
        t = np.asarray(t, dtype=float)
        base = (V + t) / self.eta
        base_clamped = np.where(base <= 0.0, 1e-10, base)
        return (self.beta / self.eta) * base_clamped ** (self.beta - 1) * self.sf(t, V)

    def cdf(self, t: np.ndarray, V: float) -> np.ndarray:
        """Cumulative distribution function F(t | V)"""
        return 1.0 - self.sf(t, V)

    def hazard(self, t: np.ndarray, V: float) -> np.ndarray:
        """Hazard rate h(t | V)"""
        t = np.asarray(t, dtype=float)
        base = (V + t) / self.eta
        base_clamped = np.where(base <= 0.0, 1e-10, base)
        return (self.beta / self.eta) * base_clamped ** (self.beta - 1)

    def mean(self, V: float) -> float:
        """MTBF given virtual age V (analytical, via upper incomplete gamma)."""
        a = 1.0 / self.beta
        x = (float(V) / self.eta) ** self.beta
        return (self.eta / self.beta) * _safe_exp_gamma(a, x)

    def variance(self, V: float) -> float:
        """Variance of time-to-failure given virtual age V (analytical)."""
        mtbf = self.mean(V)
        x = (float(V) / self.eta) ** self.beta
        a1 = 1.0 / self.beta
        a2 = 2.0 / self.beta
        term1 = self.eta * _safe_exp_gamma(a2, x)
        term2 = float(V) * _safe_exp_gamma(a1, x)
        E2 = (2.0 * self.eta / self.beta) * (term1 - term2)
        return float(E2 - mtbf**2)

    def std(self, V: float) -> float:
        """Standard deviation of time-to-failure given virtual age V."""
        var = self.variance(V)
        return np.sqrt(var) if var > 0.0 else 0.0

    def auc_improvement(self, Vn: float, t_max: float) -> float:
        """Compare Kijima reliability AUC vs pure Weibull (ar=ap=1)."""
        t = np.linspace(0, t_max, 200)
        R_k = self.sf(t, Vn)
        R_w = np.exp(-((t / self.eta) ** self.beta))
        auc_k = np.trapezoid(R_k, t)
        auc_w = np.trapezoid(R_w, t)
        if auc_w == 0.0:
            return 0.0
        return (auc_k - auc_w) / auc_w

    def mean_residual_life(self, V: float) -> float:
        """
        Expected remaining life at virtual age V via numerical integration.
        Alternative to the analytical mean() — useful for validation.
        """
        from scipy.integrate import quad
        res, _ = quad(lambda t: t * self.pdf(t, V), 0, np.inf)
        return res

    def ks_test_pit(self, x: np.ndarray, delta: np.ndarray) -> tuple[float, float]:
        """Conditional PIT Kolmogorov-Smirnov test."""
        V = self.virtual_age(x, delta)
        V_prev = np.insert(V[:-1], 0, 0.0)
        S = np.exp((V_prev / self.eta) ** self.beta - ((V_prev + x) / self.eta) ** self.beta)
        F = 1.0 - S
        F_fail = F[delta == 1]
        if len(F_fail) == 0:
            return 0.0, 1.0
        res = kstest(F_fail, "uniform")
        return float(res.statistic), float(res.pvalue)

    def r2_mrl(self, x: np.ndarray, delta: np.ndarray) -> float:
        """R² Mean Residual Life goodness-of-fit metric."""
        V = self.virtual_age(x, delta)
        V_prev = np.insert(V[:-1], 0, 0.0)
        y_pred = np.array([self.mean(v) for v in V_prev])
        y_obs = np.asarray(x)
        ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
        ss_res = np.sum((y_obs - y_pred) ** 2)
        return 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else 0.0

    def virtual_age_ratio(self, x: np.ndarray, delta: np.ndarray) -> float:
        """Average V_i / T_i ratio over the observation history."""
        V = self.virtual_age(x, delta)
        T = np.cumsum(x)
        return float(np.mean(V / T))

    def calculate_curves(self, x: np.ndarray, delta: np.ndarray, t_grid: np.ndarray = None) -> dict:
        """
        Calculate R(t), pdf(t), hazard(t) and virtual age arrays over a time grid.
        Uses Numba for acceleration.
        """
        V = self.virtual_age(x, delta)
        V_full = np.insert(V, 0, 0.0)

        try:
            mtbf = self.mean(V[-1])
            if np.isnan(mtbf) or np.isinf(mtbf) or mtbf <= 0:
                mtbf = float(np.mean(x))
        except Exception:
            mtbf = float(np.mean(x))

        future_time = 1.5 * mtbf

        t, R, h, f, V_curve = _reconstruct_grp_curves(
            x, delta, V_full, self.beta, self.eta, 5, future_time, 50
        )
        T = np.insert(np.cumsum(x), 0, 0.0)

        return {
            "t": t,
            "R": R,
            "failure_rate": h,
            "pdf": f,
            "V_curve": V_curve,
            "V": V,
            "T": T,
        }


class KijimaModelI(KijimaModel):
    """Kijima Model I: V_i = V_{i-1} + a_i * x_i"""
    def __init__(self, beta: float, eta: float, ar: float, ap: float):
        super().__init__(beta, eta, ar, ap)
        self.model_type = 1
        self.model_name = "Kijima I"

    def virtual_age(self, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return _calculate_ki(x, delta, self.ar, self.ap)


class KijimaModelII(KijimaModel):
    """Kijima Model II: V_i = a * (V_{i-1} + x_i)"""
    def __init__(self, beta: float, eta: float, ar: float, ap: float):
        super().__init__(beta, eta, ar, ap)
        self.model_type = 2
        self.model_name = "Kijima II"

    def virtual_age(self, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return _calculate_k2(x, delta, self.ar, self.ap)


class KijimaModelITD(KijimaModel):
    """Kijima Model I TD: V_i = V_{i-1} + q_i(T_i) * x_i"""
    def __init__(self, beta: float, eta: float, ar: float, ap: float, br: float, bp: float):
        super().__init__(beta, eta, ar, ap)
        self.br = float(br)
        self.bp = float(bp)
        self.model_type = 3
        self.model_name = "Kijima I TD"

    def virtual_age(self, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return _calculate_ki_td(x, delta, self.ar, self.ap, self.br, self.bp)


class KijimaModelIITD(KijimaModel):
    """Kijima Model II TD: V_i = q_i(T_i) * (V_{i-1} + x_i)"""
    def __init__(self, beta: float, eta: float, ar: float, ap: float, br: float, bp: float):
        super().__init__(beta, eta, ar, ap)
        self.br = float(br)
        self.bp = float(bp)
        self.model_type = 4
        self.model_name = "Kijima II TD"

    def virtual_age(self, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return _calculate_k2_td(x, delta, self.ar, self.ap, self.br, self.bp)


class KijimaModelITD2(KijimaModel):
    """Kijima Model I TD2: V_i = V_{i-1} + q_i(T_i) * x_i"""
    def __init__(self, beta: float, eta: float, ar: float, ap: float, br: float, bp: float):
        super().__init__(beta, eta, ar, ap)
        self.br = float(br)
        self.bp = float(bp)
        self.model_type = 5
        self.model_name = "Kijima I TD2"

    def virtual_age(self, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return _calculate_ki_td2(x, delta, self.ar, self.ap, self.br, self.bp)


class KijimaModelIITD2(KijimaModel):
    """Kijima Model II TD2: V_i = q_i(T_i) * (V_{i-1} + x_i)"""
    def __init__(self, beta: float, eta: float, ar: float, ap: float, br: float, bp: float):
        super().__init__(beta, eta, ar, ap)
        self.br = float(br)
        self.bp = float(bp)
        self.model_type = 6
        self.model_name = "Kijima II TD2"

    def virtual_age(self, x: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return _calculate_k2_td2(x, delta, self.ar, self.ap, self.br, self.bp)
