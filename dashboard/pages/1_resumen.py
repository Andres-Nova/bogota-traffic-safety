"""
Página 1 — Resumen Ejecutivo

KPIs globales, tendencia anual y distribución por gravedad y localidad.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from estilo import aplicar_estilo
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils import cargar_datos, sidebar_filtros, COLORES_GRAVEDAD, ORDEN_GRAVEDAD, MESES_ES

st.set_page_config(page_title="Resumen — Siniestralidad Bogotá", page_icon="📊", layout="wide")
aplicar_estilo()
st.title("📊 Resumen Ejecutivo")
st.caption("Bogotá D.C. · 2015–2021 · Fuente: SDM — Datos Abiertos Bogotá")

df_base = cargar_datos()
df = sidebar_filtros(df_base)

if df.empty:
    st.warning("Sin datos con los filtros seleccionados.")
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
total      = len(df)
fatales    = (df["GRAVEDAD"] == "Con Muertos").sum()
heridos    = (df["GRAVEDAD"] == "Con Heridos").sum()
solo_danos = (df["GRAVEDAD"] == "Solo Danos").sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total siniestros",   f"{total:,}")
col2.metric("Con muertos",        f"{fatales:,}",     delta=f"{fatales/total*100:.1f}%",    delta_color="inverse")
col3.metric("Con heridos",        f"{heridos:,}",     delta=f"{heridos/total*100:.1f}%",    delta_color="inverse")
col4.metric("Solo daños",         f"{solo_danos:,}",  delta=f"{solo_danos/total*100:.1f}%", delta_color="off")

st.divider()

# ── Tendencia anual ────────────────────────────────────────────────────────────
st.subheader("Siniestros por año")

anual = (
    df.groupby(["ANIO", "GRAVEDAD"])
    .size()
    .reset_index(name="conteo")
)
anual["GRAVEDAD"] = pd.Categorical(anual["GRAVEDAD"], categories=ORDEN_GRAVEDAD, ordered=True)
anual = anual.sort_values(["ANIO", "GRAVEDAD"])

fig_anual = px.bar(
    anual,
    x="ANIO", y="conteo", color="GRAVEDAD",
    color_discrete_map=COLORES_GRAVEDAD,
    category_orders={"GRAVEDAD": ORDEN_GRAVEDAD},
    labels={"conteo": "Siniestros", "ANIO": "Año", "GRAVEDAD": "Gravedad"},
    title="Siniestros anuales por gravedad (2015–2021)",
)
fig_anual.add_annotation(
    x=2020, y=anual[anual["ANIO"] == 2020]["conteo"].sum(),
    text="↓ COVID-19<br>–32% vs 2019",
    showarrow=True, arrowhead=2, arrowcolor="#666",
    ax=60, ay=-40,
    font=dict(size=12, color="#555"),
    bgcolor="rgba(255,255,255,0.8)",
)
fig_anual.update_layout(
    legend_title_text="Gravedad",
    xaxis=dict(tickmode="linear", dtick=1),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_anual, use_container_width=True)

col_grav, col_loc = st.columns(2)

# ── Distribución por gravedad ──────────────────────────────────────────────────
with col_grav:
    st.subheader("Distribución por gravedad")
    grav_conteo = (
        df["GRAVEDAD"]
        .value_counts()
        .reset_index()
        .rename(columns={"GRAVEDAD": "Gravedad", "count": "Siniestros"})
    )
    fig_grav = px.pie(
        grav_conteo, names="Gravedad", values="Siniestros",
        color="Gravedad",
        color_discrete_map=COLORES_GRAVEDAD,
        hole=0.45,
    )
    fig_grav.update_traces(textinfo="percent+label")
    fig_grav.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_grav, use_container_width=True)

# ── Top localidades ────────────────────────────────────────────────────────────
with col_loc:
    st.subheader("Localidades más peligrosas (top 10)")
    loc_conteo = (
        df.groupby("LOCALIDAD")
        .size()
        .sort_values(ascending=True)
        .tail(10)
        .reset_index(name="Siniestros")
    )
    fig_loc = px.bar(
        loc_conteo,
        x="Siniestros", y="LOCALIDAD",
        orientation="h",
        color="Siniestros",
        color_continuous_scale="Reds",
        labels={"LOCALIDAD": "Localidad"},
    )
    fig_loc.update_layout(
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_loc, use_container_width=True)

# ── Clase de accidente ─────────────────────────────────────────────────────────
st.subheader("Tipo de accidente")
clase = (
    df.groupby(["CLASE_ACC", "GRAVEDAD"])
    .size()
    .reset_index(name="Siniestros")
)
clase["GRAVEDAD"] = pd.Categorical(clase["GRAVEDAD"], categories=ORDEN_GRAVEDAD, ordered=True)
clase_total = clase.groupby("CLASE_ACC")["Siniestros"].sum().sort_values(ascending=False)
clase["CLASE_ACC"] = pd.Categorical(
    clase["CLASE_ACC"],
    categories=clase_total.index.tolist(),
    ordered=True
)

fig_clase = px.bar(
    clase.sort_values(["CLASE_ACC", "GRAVEDAD"]),
    x="CLASE_ACC", y="Siniestros", color="GRAVEDAD",
    color_discrete_map=COLORES_GRAVEDAD,
    category_orders={"GRAVEDAD": ORDEN_GRAVEDAD},
    labels={"CLASE_ACC": "Clase de accidente"},
    title="Siniestros por tipo de accidente",
)
fig_clase.update_layout(
    legend_title_text="Gravedad",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_clase, use_container_width=True)
