import pandas as pd
import streamlit as st
from app.data_processing import tratamiento, tipo_equipo, tbf
from app.fitting import fit, fit_kijima
from app.plotting import plot_pdf, plot_cdf, plot_comparison_kijima_trad, plot_comparison_failure_kijima_trad

if "file_expanded" not in st.session_state:
    st.session_state["file_expanded"] = True
st.set_page_config(
    page_title="Análisis de Confiabilidad",
    layout="wide"
)
def streamlit_app() -> None:
    st.title("Análisis de Confiabilidad")

    # Sidebar: carga de archivo
    if "file_expanded" not in st.session_state:
        st.session_state["file_expanded"] = True
    with st.sidebar.expander("Sube un archivo CSV", expanded=st.session_state.file_expanded):
        uploaded_file = st.file_uploader("", type="csv")
        if uploaded_file is not None:
            st.session_state.file_expanded = False

    if uploaded_file is None:
        return

    # Lectura y preprocesamiento
    df = pd.read_csv(uploaded_file, sep=';', decimal=",", encoding='latin-1')
    repo = tratamiento(df)

    # Selección de equipo y rango de fechas
    equipos = repo.Equipo.unique()
    equipo = st.sidebar.selectbox("Selecciona el equipo:", equipos)

    #filtro de fechas
    default_start_date = repo['Fecha_Inicio'].min().date()
    default_end_date = repo['Fecha_Fin'].max().date()
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Fecha desde", default_start_date)
    with col2:
        end_date = st.date_input("Fecha hasta", default_end_date)

    repo = repo[
        (repo['Fecha_Inicio'] >= pd.to_datetime(start_date)) &
        (repo['Fecha_Fin']   <= pd.to_datetime(end_date))
    ]

    # Selección de tipos de evento
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

    if set(mdf_ajustar) == set(mdf_censurar):
        st.sidebar.warning("No puedes censurar todos los modos de falla que ajustas.")
        return

    df_tbf_mdf = df_tbf[df_tbf['mdf'].isin(mdf_ajustar)]

    # Mostrar tabla de eventos
    with st.expander(f"Eventos {equipo}", expanded=False):
        st.dataframe(df_tbf_mdf, use_container_width=True)

    # --- Ajustes ---
    # 1) Tradicional para TBX
    result_tbx = fit(df_tbf_mdf, 'TBX', mdf_censurar)
    best_dist_tbx = result_tbx['best_distribution']
    name_tbx      = result_tbx['name']
    params_tbx    = result_tbx['parameters']
    mean_tbx      = result_tbx['mean']
    std_dev_tbx   = result_tbx['std_dev']
    aic_tbx       = result_tbx['AICc']
    bic_tbx       = result_tbx['BIC']
    # 2) Kijima I & II para TBX
    kijima_tbx = fit_kijima(df_tbf_mdf, 'TBX', mdf_censurar, modelos=[1, 2])
    # 3) Tradicional para TTX (sin Kijima)
    result_tbx = fit(df_tbf_mdf, 'TTX', mdf_censurar)
    best_dist_ttx = result_tbx['best_distribution']
    name_ttx      = result_tbx['name']
    params_ttx    = result_tbx['parameters']
    mean_ttx      = result_tbx['mean']
    std_dev_ttx   = result_tbx['std_dev']
    aic_ttx       = result_tbx['AICc']
    bic_ttx       = result_tbx['BIC']

    # Resultados Ajuste
    st.subheader("Resultados Ajuste")
    col1, col2 = st.columns(2)

    # Resultados Ajuste
    st.subheader("Resultados Ajuste")
    col1, col2 = st.columns(2)

    # Columna 1: TBX Tradicional vs Kijima
    with col1:
        subcol_trad, subcol_kijima = st.columns(2)
        
        # TBX Tradicional
        with subcol_trad.expander("TBX Tradicional", expanded=True):
            st.write(f"**- Distribución: {name_tbx}**")
            st.write("**- Parámetros:**", params_tbx)
            st.write(f"**- MTBX: {mean_tbx:.4f}**")
            st.write(f"**- Desviación Estándar: {std_dev_tbx:.4f}**")
            st.write(f"**- AIC: {aic_tbx:.4f},   BIC: {bic_tbx:.4f}**")

        # TBX Kijima I & II
        with subcol_kijima.expander("TBX Kijima I & II", expanded=True):
            for res in kijima_tbx:   # kijima_tbx = fit_kijima(..., modelos=[1,2])
                st.markdown(f"**{res['model_name']}**")
                st.write(f"**- β: {res['beta']:.4f}   η: {res['eta']:.4f}**")
                st.write(f"**- ar: {res['ar']:.4f}   ap: {res['ap']:.4f}**")
                # Aqui usamos las claves 'media' y 'desviacion estandar'
                st.write(f"**- MTBX: {res['mean_R']:.4f}**")
                st.write(f"**- Desviación Estándar: {res['std']:.4f}**")
                st.write(f"**- AIC: {res['AIC']:.4f},   BIC: {res['BIC']:.4f}**")
                #st.write(f"**- R²: {res['R2']:.4f},   p-value: {res['p_value']:.4f}**")
                #st.write(f"**- p-value: {res['p_value']:.4f}**")
                st.write("---")

    # Columna 2: TTX Tradicional
    with col2:
        with st.expander("TTX", expanded=True):
            st.write(f"**Distribución: {name_ttx}**")
            st.write("**Parámetros:**", params_ttx)
            st.write(f"**MTTX: {mean_ttx:.4f}**")
            st.write(f"**Desviación Estándar: {std_dev_ttx:.4f}**")
            st.write(f"**AIC: {aic_ttx:.4f}**,   **BIC: {bic_ttx:.4f}**")

    # Input para graficar en dos columnas
    col_tiempo_eval, col_tiempo_max = st.columns(2)
    with col_tiempo_eval:
        tiempo_especifico = st.number_input(
            "Tiempo de evaluación (h):",
            min_value=1.0, max_value=1e10, value=(mean_tbx or 1.0)
        )
    with col_tiempo_max:
        tiempo_max = st.number_input(
            "Límite para gráficos (h):",
            min_value=1.0, max_value=1e10, value=100.0
        )

    # Métricas puntuales para TBX
    conf_tbx = best_dist_tbx.SF(tiempo_especifico, show_plot=False)
    hf_tbx   = best_dist_tbx.HF(tiempo_especifico, show_plot=False)
    st.subheader("Métricas TBX Tradicional")
    m1, m2 = st.columns(2)
    m1.metric("Confiabilidad", f"{conf_tbx:.4f}")
    m2.metric("Tasa de Falla", f"{hf_tbx:.4f}")

    # --- Gráficos comparativos TBX ---
    fig_conf = plot_comparison_kijima_trad(best_dist_tbx, kijima_tbx, tiempo_max)
    fig_hf   = plot_comparison_failure_kijima_trad(best_dist_tbx, kijima_tbx, tiempo_max)
    st.plotly_chart(fig_conf, use_container_width=True)
    st.plotly_chart(fig_hf,   use_container_width=True)

    # --- Gráficos estándar TTX ---
    st.subheader("Curvas TTX (Tradicional)")
    fig_ttx_pdf = plot_pdf(best_dist_ttx, tiempo_max)
    st.plotly_chart(fig_ttx_pdf, use_container_width=True)
    fig_ttx_cdf = plot_cdf(best_dist_ttx, tiempo_max)
    st.plotly_chart(fig_ttx_cdf, use_container_width=True)