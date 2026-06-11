"""
Object-oriented reliability analysis models.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
from reliability.Fitters import Fit_Everything
from scipy.optimize import minimize
from scipy.integrate import quad
from src.reliability_analysis.utils.logger_config import setup_logging
from src.reliability_analysis.utils.config import EXCLUDED_MODELS, KIJIMA_MODELS
from src.reliability_analysis.analysis.kijima_model import (
    calculate_ki, calculate_k2, calculate_virtual_age,
    reliability, pdf, hazard, _neg_loglik
)
from src.reliability_analysis.analysis.metrics import calculate_aic_bic, ks_test_kijima_pit

logger = setup_logging("Models")


class ReliabilityFitter:
    """
    Fits reliability distributions to data.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None, excluded_models: Optional[List[str]] = None):
        """
        Initialize the fitter.
        """
        self.data = data
        self.excluded_models = excluded_models or EXCLUDED_MODELS
        logger.info("ReliabilityFitter initialized")
    
    def fit(
        self,
        dataframe: pd.DataFrame,
        column: str,
        censored_types: List[str]
    ) -> Dict[str, Any]:
        """
        Fits distributions to the given column in the dataframe.
        """
        logger.info(f"Starting fit for {column}")
        
        # Prepare data
        data = dataframe[column].dropna()
        censored_mask = dataframe['mdf'].isin(censored_types)
        data = data[~censored_mask & (data > 0)]
        data = data.to_numpy()
        
        right_censored = dataframe[column].dropna()[censored_mask].to_numpy()
        right_censored = right_censored[right_censored > 0]
        
        # Execute fit
        if column == 'TBX' and len(right_censored) > 0:
            fit_results = Fit_Everything(
                failures=data,
                right_censored=right_censored,
                exclude=self.excluded_models,
                show_histogram_plot=False,
                show_probability_plot=False,
                show_PP_plot=False
            )
        else:
            fit_results = Fit_Everything(
                failures=data,
                exclude=self.excluded_models,
                show_histogram_plot=False,
                show_probability_plot=False,
                show_PP_plot=False
            )
        
        # Extract results
        best_dist = fit_results.best_distribution
        name = fit_results.best_distribution_name
        parameters = best_dist.parameters
        mean = getattr(best_dist, 'mean', None)
        std_dev = best_dist.standard_deviation
        
        results_df = fit_results.results
        row = results_df[results_df['Distribution'] == name] if results_df is not None else pd.DataFrame()
        aic = row['AICc'].values[0] if not row.empty and 'AICc' in row.columns else None
        bic = row['BIC'].values[0] if not row.empty and 'BIC' in row.columns else None
        
        result = {
            'best_distribution': best_dist,
            'name': name,
            'parameters': parameters,
            'mean': mean,
            'std_dev': std_dev,
            'AICc': aic,
            'BIC': bic
        }
        
        logger.info(f"Fit completed: {name}")
        return result
    
    def fit_weibull(
        self, 
        column: str = 'TTX', 
        censored_failure_types: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fits Weibull distribution to data, supporting censored data.
        """
        if self.data is None or self.data.empty:
            return None
        
        if column not in self.data.columns:
            column = 'TTX' if 'TTX' in self.data.columns else 'Days'
        if column not in self.data.columns:
            logger.error(f"Column {column} not found")
            return None
        
        # Split censored and uncensored data
        df_copy = self.data.copy()
        df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
        df_copy = df_copy[df_copy[column] > 0]
        
        failures = None
        right_censored = None
        
        if censored_failure_types and 'mdf' in df_copy.columns:
            # Uncensored data
            failures_mask = ~df_copy['mdf'].isin(censored_failure_types)
            failures = df_copy[failures_mask][column].dropna().to_numpy()
            
            # Censored data
            censored_mask = df_copy['mdf'].isin(censored_failure_types)
            right_censored = df_copy[censored_mask][column].dropna().to_numpy()
        else:
            # All data as uncensored
            failures = df_copy[column].dropna().to_numpy()
        
        if failures is None or len(failures) < 2:
            logger.error("Insufficient data for Weibull fit")
            return None
        
        from reliability.Fitters import Fit_Weibull_2P
        
        try:
            kwargs = {
                'failures': failures,
                'show_probability_plot': False,
                'show_histogram_plot': False,
                'print_results': False
            }
            if right_censored is not None and len(right_censored) > 0:
                kwargs['right_censored'] = right_censored
            
            fitter = Fit_Weibull_2P(**kwargs)
            
            return {
                'beta': float(fitter.beta),
                'eta': float(fitter.alpha),
                'ar': None,
                'ap': None,
                'aic': float(fitter.AICc) if hasattr(fitter, 'AICc') else None,
                'bic': float(fitter.BIC) if hasattr(fitter, 'BIC') else None,
                'failures_count': int(len(failures)),
                'censored_count': int(len(right_censored)) if right_censored is not None else 0
            }
        except Exception as e:
            logger.error(f"Error in Weibull fit: {str(e)}")
            return None




class KijimaFitter:
    """
    Implements Kijima I & II models.
    """
    
    def __init__(self, models: List[int] = None):
        """
        Initialize Kijima models.
        """
        self.models = models or KIJIMA_MODELS
        logger.info(f"KijimaFitter initialized with models: {self.models}")
    
    def _fit_parameters(
        self,
        x: np.ndarray,
        delta: np.ndarray,
        model_type: int
    ) -> Tuple[np.ndarray, float]:
        """
        Fit Kijima parameters using optimization.
        """
        def objective(p):
            return _neg_loglik(x, delta, p[0], p[1], p[2], p[3], model_type)
        
        bounds = [(1e-6, None), (1e-6, None), (1e-2, 0.99), (1e-2, 0.99)]
        initial = [1.0, x.mean(), 0.5, 0.7]
        
        result = minimize(objective, initial, method='L-BFGS-B', bounds=bounds)
        
        logger.debug(f"Optimization converged: {result.success}")
        return result.x, -result.fun
    
    def _process_model(
        self,
        model_type: int,
        x: np.ndarray,
        delta: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process a single Kijima model.
        """
        logger.info(f"Processing Kijima {model_type}")
        
        params, ll_max = self._fit_parameters(x, delta, model_type)
        beta, eta, ar, ap = params
        
        # Virtual age
        V = calculate_virtual_age(x, delta, ar, ap, model_type)
        V_last = V[-1]
        
        # Tests and metrics
        ks_stat, p_val = ks_test_kijima_pit(x, delta, beta, eta, ar, ap, model_type)
        aic, bic = calculate_aic_bic(ll_max, 4, x.size)
        
        # Expected MTBF
        mtbf, _ = quad(lambda t: reliability(t, V_last, beta, eta), 0, np.inf)
        E2, _ = quad(lambda t: 2 * t * reliability(t, V_last, beta, eta), 0, np.inf)
        var = E2 - mtbf**2
        std = np.sqrt(var)
        
        return {
            'model_name': f"Kijima {'I' if model_type == 1 else 'II'}",
            'beta': beta,
            'eta': eta,
            'ar': ar,
            'ap': ap,
            'AIC': aic,
            'BIC': bic,
            'p_value': p_val,
            'mean': mtbf,
            'ks_stat': ks_stat,
            'std': std
        }
    
    def fit(
        self,
        dataframe: pd.DataFrame,
        column: str,
        censored_types: List[str],
        models: Union[int, List[int]] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Fits Kijima models to data.
        """
        logger.info(f"Starting Kijima fit for {column}")
        
        # Prepare data
        df = dataframe.dropna(subset=[column, 'mdf']).copy()
        df = df[df[column] > 0]
        
        x = df[column].to_numpy(dtype=float)
        delta = (~df['mdf'].isin(censored_types)).astype(float).to_numpy()
        
        # Normalize input models
        models_list = models if isinstance(models, (list, tuple)) else [models]
        models_list = models_list or self.models
        
        results = []
        
        # Process each model
        for m in models_list:
            res = self._process_model(m, x, delta)
            
            # Calculate curves
            beta, eta, ar, ap = res['beta'], res['eta'], res['ar'], res['ap']
            V = calculate_virtual_age(x, delta, ar, ap, m)
            T = np.insert(np.cumsum(x), 0, 0.0)
            t_grid = np.linspace(0, T[-1], 300)
            
            R = np.zeros_like(t_grid)
            f = np.zeros_like(t_grid)
            h = np.zeros_like(t_grid)
            
            # Calculate by interval
            V_full = np.insert(V, 0, 0.0)
            for i in range(len(T) - 1):
                mask = (t_grid >= T[i]) & (t_grid < T[i + 1])
                t_local = t_grid[mask] - T[i]
                V_i = V_full[i]
                
                vt_eta = (V_i + t_local) / eta
                v0_eta = V_i / eta
                
                R[mask] = np.exp(v0_eta ** beta - vt_eta ** beta)
                f[mask] = (beta / eta) * (vt_eta) ** (beta - 1) * R[mask]
                h[mask] = (beta / eta) * (vt_eta) ** (beta - 1)
            
            # Last interval
            mask = t_grid >= T[-1]
            t_local = t_grid[mask] - T[-1]
            V_i = V_full[-1]
            
            vt_eta = (V_i + t_local) / eta
            v0_eta = V_i / eta
            
            R[mask] = np.exp(v0_eta ** beta - vt_eta ** beta)
            f[mask] = (beta / eta) * (vt_eta) ** (beta - 1) * R[mask]
            h[mask] = (beta / eta) * (vt_eta) ** (beta - 1)
            
            res.update({
                't': t_grid,
                'R': R,
                'failure_rate': h,
                'pdf': f
            })
            
            results.append(res)
        
        logger.info(f"Kijima fit completed: {len(results)} models")
        return results[0] if len(results) == 1 else results
