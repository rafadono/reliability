"""
Jackknife resampling for confidence intervals.

Implements Leave-One-Out (LOO) Jackknife to estimate parameter uncertainty
and confidence intervals for Weibull fitting.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Union, Any
from scipy import stats


def calculate_jackknife_ci(
    data: np.ndarray,
    statistic_func = np.mean,
    confidence: float = 0.95,
    censored: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate Jackknife confidence intervals for a statistic.
    
    Leave-One-Out approach: removes each data point, recalculates
    statistic, estimates CI from resulting distribution.
    """
    # If the user passed confidence as a positional argument (e.g. statistic_func is a float)
    if isinstance(statistic_func, float):
        confidence = statistic_func
        statistic_func = np.mean

    data = np.asarray(data)
    n = len(data)
    jack_statistics = []
    
    for i in range(n):
        data_loo = np.delete(data, i)
        cens_loo = np.delete(censored, i) if censored is not None else None
        
        try:
            stat = statistic_func(data_loo, cens_loo) if censored is not None else statistic_func(data_loo)
            jack_statistics.append(stat)
        except Exception:
            try:
                stat = statistic_func(data_loo)
                jack_statistics.append(stat)
            except Exception:
                continue
    
    if not jack_statistics:
        return {
            'point_estimate': None,
            'mean': None,
            'lower': None,
            'ci_lower': None,
            'upper': None,
            'ci_upper': None,
            'std_error': None,
            'ci_width': None
        }
    
    jack_statistics = np.array(jack_statistics)
    point_estimate = np.mean(jack_statistics)
    std_error = np.std(jack_statistics, ddof=1)
    
    alpha = 1 - confidence
    z_score = stats.norm.ppf(1 - alpha/2)
    
    margin = z_score * std_error
    
    return {
        'point_estimate': float(point_estimate),
        'mean': float(point_estimate),
        'lower': float(point_estimate - margin),
        'ci_lower': float(point_estimate - margin),
        'upper': float(point_estimate + margin),
        'ci_upper': float(point_estimate + margin),
        'std_error': float(std_error),
        'ci_width': float(2 * margin),
        'sample_size': n
    }


def calculate_multi_parameter_ci(
    data: Union[np.ndarray, Dict[str, np.ndarray]],
    fit_func = None,
    param_names: List[str] = None,
    confidence: float = 0.95,
    censored: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate Jackknife CI for multiple parameters simultaneously.
    """
    # If the user passed confidence as second argument
    if isinstance(fit_func, float):
        confidence = fit_func
        fit_func = None

    # Handle dictionary of parameters directly
    if isinstance(data, dict):
        results = {}
        for param, values in data.items():
            results[param] = calculate_jackknife_ci(values, confidence=confidence)
        return results

    if fit_func is None or param_names is None:
        raise ValueError("fit_func and param_names are required when data is not a dict")

    data = np.asarray(data)
    n = len(data)
    jack_params = {name: [] for name in param_names}
    
    for i in range(n):
        data_loo = np.delete(data, i)
        cens_loo = np.delete(censored, i) if censored is not None else None
        
        try:
            params = fit_func(data_loo, cens_loo) if censored is not None else fit_func(data_loo)
            if not isinstance(params, (tuple, list, np.ndarray)):
                params = [params]
            
            for j, name in enumerate(param_names):
                if j < len(params):
                    jack_params[name].append(params[j])
        except Exception:
            continue
    
    results = {}
    alpha = 1 - confidence
    z_score = stats.norm.ppf(1 - alpha/2)
    
    for name in param_names:
        if jack_params[name]:
            values = np.array(jack_params[name])
            mean_val = np.mean(values)
            std_err = np.std(values, ddof=1)
            margin = z_score * std_err
            
            results[name] = {
                'point_estimate': float(mean_val),
                'mean': float(mean_val),
                'lower': float(mean_val - margin),
                'ci_lower': float(mean_val - margin),
                'upper': float(mean_val + margin),
                'ci_upper': float(mean_val + margin),
                'std_error': float(std_err),
                'ci_width': float(2 * margin),
                'valid_samples': len(values)
            }
        else:
            results[name] = {
                'point_estimate': None,
                'mean': None,
                'lower': None,
                'ci_lower': None,
                'upper': None,
                'ci_upper': None,
                'std_error': None,
                'ci_width': None,
                'valid_samples': 0
            }
    
    return results


def bootstrap_ci(
    data: np.ndarray,
    statistic_func = np.mean,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Bootstrap alternative to Jackknife.
    """
    # Handle aliases
    n_bootstrap = kwargs.get('n_bootstrap', n_resamples)
    if isinstance(statistic_func, int) and 'n_bootstrap' in kwargs:
        # If called like bootstrap_ci(data, n_bootstrap=1000, confidence=0.95)
        # where statistic_func might be bound to n_bootstrap positional arg if not named
        pass
    
    # If the user passed parameters in different order or missing statistic_func
    if not callable(statistic_func):
        if isinstance(statistic_func, int):
            n_bootstrap = statistic_func
        statistic_func = np.mean

    if random_state is not None:
        np.random.seed(random_state)
    
    boot_statistics = []
    data = np.asarray(data)
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        try:
            stat = statistic_func(sample)
            boot_statistics.append(stat)
        except Exception:
            continue
    
    boot_statistics = np.array(boot_statistics)
    point_estimate = statistic_func(data)
    
    alpha = 1 - confidence
    lower_pct = (alpha/2) * 100
    upper_pct = (1 - alpha/2) * 100
    
    lower = np.percentile(boot_statistics, lower_pct)
    upper = np.percentile(boot_statistics, upper_pct)
    
    return {
        'point_estimate': float(point_estimate),
        'mean': float(point_estimate),
        'lower': float(lower),
        'ci_lower': float(lower),
        'upper': float(upper),
        'ci_upper': float(upper),
        'std_error': float(np.std(boot_statistics, ddof=1)),
        'ci_width': float(upper - lower),
        'method': 'bootstrap'
    }


def sensitivity_analysis(
    data: np.ndarray,
    fit_func,
    param_name: str,
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Analyze sensitivity of parameter estimate to data points.
    """
    data = np.asarray(data)
    n = len(data)
    influences = []
    
    point_estimate = fit_func(data)
    
    for i in range(n):
        data_loo = np.delete(data, i)
        try:
            param_loo = fit_func(data_loo)
            influence = abs(param_loo - point_estimate)
            influences.append(influence)
        except Exception:
            influences.append(0)
    
    influences = np.array(influences)
    threshold = np.mean(influences) + 2 * np.std(influences)
    outliers = np.where(influences > threshold)[0].tolist()
    
    return {
        'influence': influences.tolist(),
        'outliers': outliers,
        'threshold': float(threshold),
        'max_influence': float(np.max(influences)),
        'mean_influence': float(np.mean(influences))
    }


class JackknifeAnalyzer:
    """
    Leave-One-Out Jackknife resampling.
    """
    calculate_jackknife_ci = staticmethod(calculate_jackknife_ci)
    calculate_multi_parameter_ci = staticmethod(calculate_multi_parameter_ci)
    bootstrap_ci = staticmethod(bootstrap_ci)
    sensitivity_analysis = staticmethod(sensitivity_analysis)
