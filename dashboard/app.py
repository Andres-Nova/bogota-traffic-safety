"""
Dashboard de Siniestralidad Vial — Bogotá D.C.
Datos: Secretaría Distrital de Movilidad (SDM) — Datos Abiertos Bogotá
Período: 2015–2021 | 209,861 siniestros

Punto de entrada multipage para Streamlit Cloud.
"""
import streamlit as st

st.set_page_config(
    page_title="Siniestralidad Vial Bogotá",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚦 Siniestralidad Vial — Bogotá D.C.")
st.markdown(
    """
    Análisis de **209,861 siniestros viales** registrados entre 2015 y 2021
    por la Secretaría Distrital de Movilidad, con datos del sistema SIGAT
    (Sistema de Información Geográfica de Accidentes de Tránsito).

    ---
    """
)

col1, col2, col3 = st.columns(3)
col1.page_link("pages/1_resumen.py",   label="📊 Resumen ejecutivo",       icon="📊")
col2.page_link("pages/2_mapa.py",      label="🗺️ Mapa de siniestros",      icon="🗺️")
col3.page_link("pages/3_analisis.py",  label="📈 Análisis temporal",       icon="📈")

st.markdown(
    """
    **Fuente:** [Datos Abiertos Bogotá — Histórico Siniestros](https://datosabiertos.bogota.gov.co/en/dataset/historico-siniestros-bogota-d-c)
    **API:** ArcGIS FeatureServer / SDM
    **Licencia:** Creative Commons Attribution 4.0
    """
)
