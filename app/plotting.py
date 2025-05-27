import plotly.graph_objects as go
import numpy as np

def plot_metric(best_fit, tiempo_max, metric_func, title, y_label):
    # 1) generar array de tiempos y valores de la métrica
    tiempos = np.linspace(0, tiempo_max, 100)
    metric_values = metric_func(tiempos, show_plot=False)
    
    # 2) crear figura y traza
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tiempos,
            y=metric_values,
            mode="lines",
            name=title,
            hoverinfo="x+y"             # para que muestre coord X/Y en el hover
        )
    )
    
    # 3) layout con título y etiquetas
    fig.update_layout(
        title=title,
        xaxis_title="Tiempo (h)",
        yaxis_title=y_label,
        hovermode="x unified"         # unifica el hover en la misma X
    )
    
    return fig

def plot_confiabilidad(best_fit, tiempo_max):
    return plot_metric(
        best_fit, 
        tiempo_max, 
        best_fit.SF, 
        title="Confiabilidad", 
        y_label="Confiabilidad"
    )

def plot_tasa_de_falla(best_fit, tiempo_max):
    return plot_metric(
        best_fit, 
        tiempo_max, 
        best_fit.HF, 
        title="Tasa de Falla", 
        y_label="Tasa de Falla (1/h)"
    )

def plot_pdf(best_fit, tiempo_max):
    return plot_metric(
        best_fit, 
        tiempo_max, 
        best_fit.PDF, 
        title="Densidad de Probabilidad", 
        y_label="Densidad de Probabilidad"
    )

def plot_cdf(best_fit, tiempo_max):
    return plot_metric(
        best_fit, 
        tiempo_max, 
        best_fit.CDF, 
        title="Probabilidad Acumulada", 
        y_label="Probabilidad Acumulada"
    )

def plot_comparison_kijima_trad(trad_fit, kijima_results, tiempo_max: float):
    t = np.linspace(0, tiempo_max, 200)
    fig = go.Figure()

    # Tradicional
    fig.add_trace(go.Scatter(
        x=t,
        y=trad_fit.SF(t, show_plot=False),
        name="Tradicional",
        mode="lines"
    ))

    # Kijima I & II (recortadas a tiempo_max)
    for res in kijima_results:
        mask = res['t'] <= tiempo_max
        fig.add_trace(go.Scatter(
            x=res['t'][mask],
            y=res['R'][mask],
            name=res['model_name'],
            mode="lines"
        ))

    fig.update_layout(
        title="Confiabilidad Comparativa (TBX)",
        xaxis_title="Tiempo (h)",
        yaxis_title="Confiabilidad",
        hovermode="x unified"
    )
    return fig

def plot_comparison_failure_kijima_trad(trad_fit, kijima_results, tiempo_max: float):
    t = np.linspace(0, tiempo_max, 200)
    fig = go.Figure()

    # Tradicional
    fig.add_trace(go.Scatter(
        x=t,
        y=trad_fit.HF(t, show_plot=False),
        name="Tradicional",
        mode="lines"
    ))

    # Kijima I & II (recortadas a tiempo_max)
    for res in kijima_results:
        mask = res['t'] <= tiempo_max
        fig.add_trace(go.Scatter(
            x=res['t'][mask],
            y=res['failure_rate'][mask],
            name=res['model_name'],
            mode="lines"
        ))

    fig.update_layout(
        title="Tasa de Falla Comparativa (TBX)",
        xaxis_title="Tiempo (h)",
        yaxis_title="Tasa de Falla (1/h)",
        hovermode="x unified"
    )
    return fig
