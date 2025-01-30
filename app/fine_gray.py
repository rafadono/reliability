import pandas as pd
import matplotlib.pyplot as plt

def calcular_riesgo_fine_gray(df: pd.DataFrame, columna_tiempo: str, columna_evento: str, censura_valor: int = 0):
    """
    Calcula el riesgo acumulado subdistribucional para cada modo de falla usando el método Fine-Gray.
    
    Parámetros:
    - df: DataFrame con los datos procesados.
    - columna_tiempo: Nombre de la columna que contiene los tiempos hasta el evento o censura.
    - columna_evento: Nombre de la columna que indica el tipo de evento (modo de falla).
    - censura_valor: Valor que indica censura en la columna de eventos (por defecto, 0).

    Devuelve:
    - risk_df: DataFrame con el riesgo acumulado para cada modo.
    """
    # Identificar modos de falla únicos
    modos = df[columna_evento].unique()
    modos = [m for m in modos if m != censura_valor]  # Excluir censura

    # Ordenar datos por tiempo
    df = df.sort_values(by=columna_tiempo).reset_index(drop=True)

    # Inicializar variables para riesgo acumulado
    n_at_risk = len(df)
    cumulative_risks = {m: [] for m in modos}  # Riesgos acumulados por modo
    times = []  # Tiempos de eventos

    # Iterar sobre los datos para calcular riesgos acumulados
    for i, row in df.iterrows():
        event = row[columna_evento]
        times.append(row[columna_tiempo])
        if event in modos:  # Si es un evento (no censura)
            for m in modos:
                if m == event:
                    # Sumar al riesgo acumulado del modo actual
                    cumulative_risks[m].append((1 / n_at_risk) + (cumulative_risks[m][-1] if cumulative_risks[m] else 0))
                else:
                    # Mantener el riesgo acumulado de los demás modos
                    cumulative_risks[m].append(cumulative_risks[m][-1] if cumulative_risks[m] else 0)
        else:  # Si es censura
            for m in modos:
                # Mantener los riesgos acumulados
                cumulative_risks[m].append(cumulative_risks[m][-1] if cumulative_risks[m] else 0)
        n_at_risk -= 1  # Reducir el número en riesgo

    # Crear DataFrame para riesgos acumulados
    risk_df = pd.DataFrame({"Tiempo": times})
    for m in modos:
        risk_df[f"Riesgo Acumulado Modo {m}"] = cumulative_risks[m]

    return risk_df

def graficar_riesgo_fine_gray(risk_df: pd.DataFrame, titulo: str = "Riesgo Acumulado Subdistribucional"):
    """
    Genera una gráfica de los riesgos acumulados para cada modo.

    Parámetros:
    - risk_df: DataFrame con los riesgos acumulados por modo.
    - titulo: Título de la gráfica.
    """
    plt.figure(figsize=(10, 6))
    for columna in risk_df.columns:
        if columna != "Tiempo":
            plt.step(risk_df["Tiempo"], risk_df[columna], label=columna, where="post")
    plt.xlabel("Tiempo")
    plt.ylabel("Riesgo Acumulado")
    plt.title(titulo)
    plt.legend()
    plt.grid()
    plt.show()