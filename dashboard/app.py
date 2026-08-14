"""
Dashboard de Siniestralidad Vial — Bogotá D.C.
Datos: Secretaría Distrital de Movilidad (SDM) — Datos Abiertos Bogotá
Período: 2015–2021 | 209,861 siniestros

Punto de entrada multipage para Streamlit Cloud.
"""
import streamlit as st
from estilo import mostrar_header, mostrar_footer

st.set_page_config(
    page_title="Siniestralidad Vial Bogotá",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

mostrar_header(
    titulo="Siniestralidad Vial — Bogotá D.C.",
    emoji="🚦",
    descripcion=(
        "209,861 siniestros viales (2015–2021) · "
        "Secretaría Distrital de Movilidad · LightGBM AUC 0.78"
    ),
)

col1, col2, col3 = st.columns(3)
col1.page_link("pages/1_resumen.py",  label="📊 Resumen ejecutivo",  icon="📊")
col2.page_link("pages/2_mapa.py",     label="🗺️ Mapa de siniestros", icon="🗺️")
col3.page_link("pages/3_analisis.py", label="📈 Análisis temporal",  icon="📈")

st.markdown(
    """
    **Fuente:** [Datos Abiertos Bogotá — Histórico Siniestros](https://datosabiertos.bogota.gov.co/en/dataset/historico-siniestros-bogota-d-c)
    **API:** ArcGIS FeatureServer / SDM &nbsp;·&nbsp; **Licencia:** Creative Commons Attribution 4.0
    """
)

mostrar_footer()
