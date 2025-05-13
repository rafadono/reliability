from reliability.Fitters import Fit_Everything
import pandas as pd

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
            print("censuras",right_censored)
            print("tamaño",len(right_censored)) 
            fit_results = Fit_Everything(failures=data, right_censored=right_censored, exclude=excluded_models, show_histogram_plot=False, show_probability_plot=False, show_PP_plot=False)
        else:
            print("no hay censuras")
            fit_results = Fit_Everything(failures=data, exclude=excluded_models, show_histogram_plot=False, show_probability_plot=False, show_PP_plot=False)
    else:
        fit_results = Fit_Everything(failures=data, exclude=excluded_models, show_histogram_plot=False, show_probability_plot=False, show_PP_plot=False)

    best_distribution = fit_results.best_distribution
    best_distribution_name = fit_results.best_distribution_name
    parametros = best_distribution.parameters

    promedio_distribucion = best_distribution.mean if hasattr(best_distribution, 'mean') else None

    return best_distribution, (best_distribution_name, parametros), promedio_distribucion