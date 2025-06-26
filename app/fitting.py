from reliability.Fitters import Fit_Everything
import pandas as pd
import numpy as np
from app.kijima_model import reliability, pdf, calculate_virtual_age
from typing import Union, List, Dict, Any
from app.tests import calculate_aic_bic, ks_test_kijima_pit
from app.kijima_model import _neg_loglik
from scipy.optimize import minimize
from scipy.integrate import quad

def fit(dataframe: pd.DataFrame, columna: str, tipos_censurados: list) -> tuple:
    data = dataframe[columna].dropna()
    censurados = dataframe['mdf'].isin(tipos_censurados)
    data = data[~censurados & (data > 0)]
    data = data.to_numpy()
    right_censored = dataframe[columna].dropna()[censurados].to_numpy()
    right_censored = right_censored[right_censored > 0]

    # Especificar correctamente los modelos a excluir
    excluded_models = [
        'Weibull_2P', 'Weibull_CR', 'Weibull_Mixture', 'Weibull_DS',
        'Gamma_2P', 'Loglogistic_2P', 'Gamma_3P', 'Lognormal_3P',
        'Loglogistic_3P', 'Gumbel_2P', 'Exponential_2P', 'Beta_2P'
    ]
    
    if columna == 'TBX':
        if right_censored is not None and len(right_censored) > 0:
            fit_results = Fit_Everything(failures=data, right_censored=right_censored, exclude=excluded_models, show_histogram_plot=False, show_probability_plot=False, show_PP_plot=False)
        else:
            fit_results = Fit_Everything(failures=data, exclude=excluded_models, show_histogram_plot=False, show_probability_plot=False, show_PP_plot=False)
    else:
        fit_results = Fit_Everything(failures=data, exclude=excluded_models, show_histogram_plot=False, show_probability_plot=False, show_PP_plot=False)

    best_dist        = fit_results.best_distribution
    name, parametros = fit_results.best_distribution_name, best_dist.parameters
    parametros = best_dist.parameters
    promedio         = getattr(best_dist, 'mean', None)
    std_dev = best_dist.standard_deviation

    results_df = fit_results.results
    row = results_df[results_df['Distribution'] == name]
    aic = row['AICc'].values[0]
    bic  = row['BIC'].values[0]

    #beta = getattr(best_dist, 'beta', parametros.get('beta'))
    #eta  = getattr(best_dist, 'eta', parametros.get('eta'))
    #ks_stat, p_value = ks_test_weibull_pit(data, beta, eta)


    return {
        'best_distribution': best_dist,
        'name':         name,
        'parameters':   parametros,
        'mean':         promedio,
        'std_dev':      std_dev,
        'AICc':         aic,
        'BIC':          bic
        #'KS_stat':      ks_stat,
        #'KS_pvalue':    p_value
    }

def fit_kijima(
    dataframe: pd.DataFrame,
    columna: str,
    tipos_censurados: List[str],
    modelos: Union[int, List[int]] = 1
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:

    df = dataframe.dropna(subset=[columna, 'mdf']).copy()
    df = df[df[columna] > 0]

    x = df[columna].to_numpy(dtype=float)
    delta = (~df['mdf'].isin(tipos_censurados)).astype(float).to_numpy()
    T = np.insert(np.cumsum(x), 0, 0.0)
    t_grid = np.linspace(0, T[-1], 300)

    modelos_list = modelos if isinstance(modelos, (list, tuple)) else [modelos]
    resultados = []

    for m in modelos_list:
        res = process_model(m, x, delta)
        beta, eta, ar, ap = res['beta'], res['eta'], res['ar'], res['ap']

        # Edad virtual en cada evento
        V = np.zeros_like(x)
        v_prev = 0.0
        for i in range(len(x)):
            a = ar if delta[i] == 1 else ap
            if m == 1:  # Kijima I
                V[i] = v_prev + a * x[i]
            elif m == 2:  # Kijima II
                V[i] = a * (v_prev + x[i])
            else:
                raise ValueError("Solo se admite Kijima I (1) o II (2)")
            v_prev = V[i]
        V_full = np.insert(V, 0, 0.0)
        # Inicialización de curvas
        R = np.zeros_like(t_grid)
        f = np.zeros_like(t_grid)
        h = np.zeros_like(t_grid)

        # Cálculo tramo a tramo
        for i in range(len(T) - 1):
            mask = (t_grid >= T[i]) & (t_grid < T[i + 1])
            t_local = t_grid[mask] - T[i]
            V_i = V_full[i]

            vt_eta = (V_i + t_local) / eta
            v0_eta = V_i / eta

            R[mask] = np.exp(v0_eta ** beta - vt_eta ** beta)
            f[mask] = (beta / eta) * (vt_eta) ** (beta - 1) * R[mask]
            h[mask] = (beta / eta) * (vt_eta) ** (beta - 1)

        # Último tramo (post-falla)
        mask = t_grid >= T[-1]
        t_local = t_grid[mask] - T[-1]
        V_i = V_full[-1]

        vt_eta = (V_i + t_local) / eta
        v0_eta = V_i / eta

        R[mask] = np.exp(v0_eta ** beta - vt_eta ** beta)
        f[mask] = (beta / eta) * (vt_eta) ** (beta - 1) * R[mask]
        h[mask] = (beta / eta) * (vt_eta) ** (beta - 1)
        resultados.append({
            'model_name':   res['modelo'],
            'beta':         beta,
            'eta':          eta,
            'ar':           ar,
            'ap':           ap,
            'AIC':          res['AIC'],
            'BIC':          res['BIC'],
            #'p_value':      res['p_value'],
            #'R2':           res['R^2'],
            'mean_R':       res['media'],
            't':            t_grid,
            'R':            R,
            'failure_rate': h,
            #'V_i':          V_i,
            'std':          res['std']
        })

    return resultados[0] if len(resultados) == 1 else resultados

def process_model(model_type, x, delta):
    params, llmax = fit_parameters(x, delta, model_type)
    beta, eta, ar, ap = params
    V = calculate_virtual_age(x, delta, ar, ap, model_type)
    V_last = V[-1]  # Edad virtual al último evento
    #ks_stat, p_val = kolmogorov_smirnov_test(x, beta, eta)
    ks_stat, p_val = ks_test_kijima_pit(x, delta, beta, eta, ar, ap, model_type)
    aic, bic = calculate_aic_bic(llmax, 4, x.size)
    # MTBF esperado a partir del último V
    mtbf, _ = quad(lambda t: reliability(t, V_last, beta, eta), 0, np.inf)
    E2, _   = quad(lambda t: 2 * t * reliability(t, V_last, beta, eta), 0, np.inf)
    var = E2 - mtbf**2
    std = np.sqrt(var)
    return {
        'modelo': f"Kijima {'I' if model_type==1 else 'II'}",
        'beta': beta, 
        'eta': eta, 
        'ar': ar, 
        'ap': ap,
        'AIC': aic, 
        'BIC': bic, 
        'p_value': p_val,
        'media': mtbf, 
        'kolmogorov-smirnov': ks_stat,
        #'R^2': calculate_r2(x, V),
        #'R^2': r2,
        "std": std
    }

# Interfaz principal
def calculate_kijima_values(TBX, Type, Valid, model):
    # 1) Prepara x y delta
    x     = np.asarray(TBX, float)
    delta = np.asarray([{'MC':1,'MCE':1,'MP':0}.get(t,0) for t in Type], float)

    # 2) Normaliza la entrada de modelo(s) a lista
    model_list = model if isinstance(model, (list, tuple)) else [model]

    results = []
    for m in model_list:
        # Ajuste de parámetros y métricas
        res = process_model(m, x, delta)
        beta, eta, ar, ap = res['beta'], res['eta'], res['ar'], res['ap']

        # 3) Recalcula edad virtual y curvas en t estándar
        V = calculate_virtual_age(x, delta, ar, ap, m)
        t = np.linspace(0, 100, 100)
        R = reliability(t, V[-1], beta, eta)
        failure_rate = pdf(t, V[-1], beta, eta) / R

        # 4) Ensambla salida con los mismos nombres que en tu versión “vieja”
        data = {
            "R": R,
            "V": V,
            "mean_r": res['mean_R'],
            "std_r": res['std_R'],
            "p_value": res['p_value'],
            "aic_value": res['AIC'],
            "bic_value": res['BIC'],
            "model_name": "Kijima I" if m == 1 else "Kijima II",
            "failure_rate": failure_rate,
            "beta": beta,
            "eta": eta,
            "ar": ar,
            "ap": ap,
            "std": res['std']
        }
        results.append(data)

    # 5) Si solo pediste un modelo, devuelve un dict en lugar de lista
    return results[0] if len(results) == 1 else results

def fit_parameters(x, delta, model_type):
    # envuelve _neg_loglik_jit para SciPy
    def obj(p):
        return _neg_loglik(x, delta, p[0], p[1], p[2], p[3], model_type)
    bounds = [(1e-6,None),(1e-6,None),(1e-2,0.99),(1e-2,0.99)]
    init = [1.0, x.mean(), 0.5, 0.7]
    res = minimize(obj, init, method='L-BFGS-B', bounds=bounds)
    return res.x, -res.fun