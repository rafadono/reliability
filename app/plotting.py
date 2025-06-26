import plotly.graph_objects as go
import numpy as np
from app.kijima_model import calculate_virtual_age

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
def plot_repair_efficiency(
    tbx: np.ndarray,
    deltas: np.ndarray,
    kijima_results: list[dict],
    max_marker_diameter: float = 40.0
) -> go.Figure:
    """
    Métrica unificada de eficiencia basada en cambio simétrico de edad virtual:
      E_i = (V_{i-1} - V_i) / (V_{i-1} + V_i)
    - Para Kijima I, V_i >= V_{i-1} ⇒ E_i ≤ 0 (deterioro)
    - Para Kijima II, V_i ≤ V_{i-1} ⇒ E_i ≥ 0 (mejora)
    Este indicador está acotado en [-1,1], donde 1 = mejora máxima, -1 = peor deterioro,
    y 0 = sin cambio.
    """
    fig = go.Figure()
    for res in kijima_results:
        model_name = res.get('model_name', '')
        mt = 2 if 'II' in model_name else 1
        ar, ap = res['ar'], res['ap']

        x = np.asarray(tbx, dtype=float)
        mask = x > 0
        x = x[mask]
        delta = deltas[mask]

        # Calcular edad virtual para cada modelo
        V = calculate_virtual_age(x, delta, ar, ap, mt)
        V_prev = np.concatenate(([0.0], V[:-1]))

        # Calcular eficiencia simétrica
        sum_V = V_prev + V
        diff_V = V_prev - V
        # Evitar división por cero
        E = np.where(sum_V != 0, diff_V / sum_V, 0.0)

        # Tiempo acumulado
        T = np.cumsum(x)
        sizeref = 2.0 * x.max() / (max_marker_diameter**2)

        fig.add_trace(go.Scatter(
            x=T, y=E, mode='markers+lines', name=model_name,
            marker=dict(size=x, sizemode='area', sizeref=sizeref,
                        opacity=0.6, line=dict(width=1, color='DarkSlateGrey'))
        ))

    fig.update_layout(
        title="Eficiencia simétrica de reparación vs Tiempo",
        xaxis_title="Tiempo acumulado (h)",
        yaxis_title="E_i = (V_prev - V_i)/(V_prev + V_i) ∈ [-1,1]",
        yaxis=dict(range=[-1,1]), hovermode='x unified'
    )
    return fig

def plot_reliability_efficiency(
    kijima_results: list[dict]
) -> go.Figure:
    """
    Grafica la eficiencia de confiabilidad usando la diferencia desfasada:
    E_i = (R[i] - R[i-1]) / (1 - R[i-1]), con E[0] = 1.

    - Cada res debe contener:
      - 't': vector de tiempos
      - 'R': vector de confiabilidades en esos tiempos
    """
    fig = go.Figure()
    for res in kijima_results:
        t = np.asarray(res['t'], dtype=float)
        R = np.asarray(res['R'], dtype=float)
        E = np.zeros_like(R)
        if R.size > 0:
            E[0] = 1.0
        if R.size > 1:
            denom = 1.0 - R[:-1]
            diff = R[1:] - R[:-1]
            E[1:] = np.where(denom > 0, diff / denom, 0.0)
        E = np.clip(E, 0.0, 1.0)
        fig.add_trace(go.Scatter(
            x=t, y=E, mode='lines', name=res['model_name']
        ))
    fig.update_layout(
        title="Eficiencia de confiabilidad por diferencia desfasada",
        xaxis_title="Tiempo (h)",
        yaxis_title="E_i = (R[i] - R[i-1])/(1 - R[i-1])",
        yaxis=dict(range=[0,1]),
        hovermode='x unified'
    )
    return fig
