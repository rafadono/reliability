import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from app.data_processing import tratamiento, tipo_equipo, tbf, calcular_valores
from app.fitting import fit

def plot_metric(best_fit, tiempo_max, metric_func, title, y_label):
    tiempos = np.linspace(0, tiempo_max, 100)
    metric_values = metric_func(tiempos, show_plot=False)

    plt.figure(figsize=(10, 6))
    plt.plot(tiempos, metric_values)
    plt.title(title)
    plt.xlabel('Tiempo (h)')
    plt.ylabel(y_label)
    plt.grid(True)

    fig = plt.gcf()
    plt.close(fig)
    return fig

def plot_confiabilidad(best_fit, tiempo_max):
    return plot_metric(best_fit, tiempo_max, best_fit.SF, "Confiabilidad", "Confiabilidad")

def plot_tasa_de_falla(best_fit, tiempo_max):
    return plot_metric(best_fit, tiempo_max, best_fit.HF, "Tasa de Falla", "Tasa de Falla (1/h)")

def plot_pdf(best_fit, tiempo_max):
    return plot_metric(best_fit, tiempo_max, best_fit.PDF, "Densidad de Probabilidad", "Densidad de Probabilidad")

def plot_cdf(best_fit, tiempo_max):
    return plot_metric(best_fit, tiempo_max, best_fit.CDF, "Probabilidad Acumulada", "Probabilidad Acumulada")

if "file_expanded" not in st.session_state:
    st.session_state["file_expanded"] = True

def streamlit_app()-> None:
    st.title("Análisis de Confiabilidad")

    if "file_expanded" not in st.session_state:
        st.session_state["file_expanded"] = True
        
    with st.sidebar.expander("Sube un archivo CSV", expanded=st.session_state.file_expanded):
        uploaded_file = st.file_uploader("", type="csv")
        # Cambia el estado para colapsar el expander al cargar un archivo
        if uploaded_file is not None:
            st.session_state.file_expanded = False

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=';', decimal=",", encoding='latin-1')
        repo = tratamiento(df)

        lista_equipos = repo.Equipo.unique()
        equipo = st.sidebar.selectbox("Selecciona el equipo con el desea trabajar:", lista_equipos)

        #filtro de fechas
        default_start_date = repo['Fecha_Inicio'].min().date()
        default_end_date = repo['Fecha_Fin'].max().date()

        col1, col2 = st.sidebar.columns(2)

        with col1:
            start_date = st.date_input("Fecha desde", default_start_date)

        with col2:
            end_date = st.date_input("Fecha hasta", default_end_date)

        repo = repo[(repo['Fecha_Inicio'] >= pd.to_datetime(start_date)) & (repo['Fecha_Fin'] <= pd.to_datetime(end_date))]

        lista_tipos = tipo_equipo(repo, equipo)
        tipos = tuple(lista_tipos)

        lista_det = st.sidebar.multiselect(
            "Selecciona los tipos a ajustar:",
            tipos,
            tipos
        )

        lista_censura = st.sidebar.multiselect(
            "Selecciona los tipos a censurar:",
            lista_det,
            lista_det
        )

        df_tbf = tbf(repo, equipo, lista_det)
        df_tbf_tipos = df_tbf[df_tbf['Tipo'].isin(lista_det)]
        
        lista_mdf_ajustar = df_tbf_tipos['mdf'].unique()

        mdf_ajustar = st.sidebar.multiselect(
            "Selecciona los modos de falla a ajustar:",
            lista_mdf_ajustar,
            lista_mdf_ajustar
        )

        df_tbf_tipos_censura = df_tbf_tipos[df_tbf_tipos['Tipo'].isin(lista_censura)]

        lista_mdf_censurar = df_tbf_tipos_censura[df_tbf_tipos_censura['mdf'].isin(mdf_ajustar)]['mdf'].unique()

        lista_mdf_censurar_fin = list(set(lista_mdf_ajustar) & set(lista_mdf_censurar))

        mdf_censurar = st.sidebar.multiselect(
            "Selecciona los modos de falla a censurar:",
            lista_mdf_censurar_fin,
            lista_mdf_censurar_fin
        )

        mdf_ajustar_set = set(mdf_ajustar)
        mdf_censurar_set = set(mdf_censurar)

        # Verifica si todos los modos a ajustar están también en los modos censurados
        if mdf_ajustar_set == mdf_censurar_set:
            st.sidebar.warning("No se pueden censurar todos los modos de falla seleccionados para ajustar.")
            return
        
        df_tbf_mdf = df_tbf_tipos[df_tbf_tipos['mdf'].isin(mdf_ajustar)]

        df_equipo = df_tbf_mdf[df_tbf_mdf['Equipo'] == equipo].copy()
        df_equipo.drop(columns=['Equipo'], inplace=True)

        with st.expander(f"Tabla de Eventos {equipo}", expanded=False):
            st.dataframe(df_equipo)

        fit_tbx, params_tbx, promedio_tbx = fit(df_tbf_mdf, 'TBX', mdf_censurar)
        fit_ttx, params_ttx, promedio_ttx = fit(df_tbf_mdf, 'TTX', mdf_censurar)

        if fit_tbx and fit_ttx:
            st.subheader("Distribución y Parámetros")

            col1, col2 = st.columns(2)

            with col1.expander("Resultados para TBX", expanded=True):
                st.write(f"**Distribución**: {params_tbx[0]}")
                st.write("**Parámetros:**")
                if isinstance(params_tbx[1], dict):  # Si los parámetros son un diccionario con nombres
                    parametros_formateados = ", ".join(f"{valor:.4f}" for valor in params_tbx[1].values())
                else:  # Si los parámetros son una lista sin nombres
                    parametros_formateados = ", ".join(f"{valor:.4f}" for valor in params_tbx[1])
                st.write(parametros_formateados)
                st.write(f"**MTBX**: {promedio_tbx:.4f}")

            with col2.expander("Resultados para TTX", expanded=True):
                st.write(f"**Distribución**: {params_ttx[0]}")
                st.write("**Parámetros:**")
                if isinstance(params_ttx[1], dict):  # Si los parámetros son un diccionario con nombres
                    parametros_formateados = ", ".join(f"{valor:.4f}" for valor in params_ttx[1].values())
                else:  # Si los parámetros son una lista sin nombres
                    parametros_formateados = ", ".join(f"{valor:.4f}" for valor in params_ttx[1])
                st.write(parametros_formateados)
                st.write(f"**MTTX**: {promedio_ttx:.4f}")

            tiempo_especifico = st.number_input(
                "Ingresa tiempo de evaluación (h):", 
                min_value=0.0, 
                max_value=1000000.0, 
                value=promedio_tbx if promedio_tbx is not None else 1.0
            )

            if tiempo_especifico > 0:

                confiabilidad_tbx, tasa_falla_tbx, _, _ = calcular_valores(fit_tbx, tiempo_especifico, 'TBX')
                #_, _, densidad_ttx, probabilidad_ttx = calcular_valores(fit_ttx, tiempo_especifico, 'TTX')

                #col1, col2, col3, col4 = st.columns(2)
                col1, col2 = st.columns(2)
                col1.metric("Confiabilidad", f"{confiabilidad_tbx:.4f}")
                col2.metric("Tasa de Falla", f"{tasa_falla_tbx:.4f}")
                #col3.metric("Densidad de Probabilidad ", f"{densidad_ttx:.4f}")
                #col4.metric("Probabilidad Acumulada (TTX)", f"{probabilidad_ttx:.4f}")

            tiempo_max = st.number_input("Ingresa límite para los gráficos (h):", min_value=1.0, max_value=1000000.0, value=100.0)

            conf_tbx = plot_confiabilidad(fit_tbx, tiempo_max)
            falla_tbx = plot_tasa_de_falla(fit_tbx, tiempo_max)

            pdf_ttx = plot_pdf(fit_ttx, tiempo_max)
            cdf_ttx = plot_cdf(fit_ttx, tiempo_max)

            st.pyplot(conf_tbx)
            st.pyplot(falla_tbx)
            st.pyplot(pdf_ttx)
            st.pyplot(cdf_ttx)