"""
Página 3 — Análisis Temporal y Estacional

Distribución por mes, día de semana, comparación anual y tablas detalladas.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from estilo import aplicar_estilo, toggle_tema_sidebar, aplicar_tema_fig
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils import (
    cargar_datos, sidebar_filtros,
    COLORES_GRAVEDAD, ORDEN_GRAVEDAD, DIAS_ORDEN, MESES_ES,
)

st.set_page_config(page_title="Análisis — Siniestralidad Bogotá", page_icon="📈", layout="wide")
toggle_tema_sidebar()
aplicar_estilo()
st.title("📈 Análisis Temporal y Estacional")
st.caption("Bogotá D.C. · 2015–2021 · Fuente: SDM — Datos Abiertos Bogotá")

df_base = cargar_datos()
df = sidebar_filtros(df_base)

if df.empty:
    st.warning("Sin datos con los filtros seleccionados.")
    st.stop()

# ── Por día de semana ──────────────────────────────────────────────────────────
st.subheader("¿Qué día de la semana son más frecuentes los siniestros?")

col_dias, col_mes = st.columns(2)

with col_dias:
    dias_grav = (
        df.groupby(["DIA_SEMANA_ES", "GRAVEDAD"])
        .size()
        .reset_index(name="Siniestros")
    )
    dias_grav["DIA_SEMANA_ES"] = pd.Categorical(
        dias_grav["DIA_SEMANA_ES"], categories=DIAS_ORDEN, ordered=True
    )
    dias_grav["GRAVEDAD"] = pd.Categorical(
        dias_grav["GRAVEDAD"], categories=ORDEN_GRAVEDAD, ordered=True
    )
    dias_grav = dias_grav.sort_values(["DIA_SEMANA_ES", "GRAVEDAD"])

    fig_dias = px.bar(
        dias_grav,
        x="DIA_SEMANA_ES", y="Siniestros", color="GRAVEDAD",
        color_discrete_map=COLORES_GRAVEDAD,
        category_orders={"GRAVEDAD": ORDEN_GRAVEDAD, "DIA_SEMANA_ES": DIAS_ORDEN},
        labels={"DIA_SEMANA_ES": "Día"},
        title="Siniestros por día de semana",
    )
    fig_dias.update_layout(legend_title_text="Gravedad")
    aplicar_tema_fig(fig_dias)
    st.plotly_chart(fig_dias, use_container_width=True)
    st.caption(
        "Jueves y lunes concentran más siniestros — días laborales de alta movilidad. "
        "El sábado es el día con menos accidentes."
    )

# ── Por mes ────────────────────────────────────────────────────────────────────
with col_mes:
    mes_grav = (
        df.groupby(["MES", "GRAVEDAD"])
        .size()
        .reset_index(name="Siniestros")
    )
    mes_grav["MES_ES"] = mes_grav["MES"].map(MESES_ES)
    mes_grav["GRAVEDAD"] = pd.Categorical(
        mes_grav["GRAVEDAD"], categories=ORDEN_GRAVEDAD, ordered=True
    )
    meses_orden = [MESES_ES[m] for m in range(1, 13)]

    fig_mes = px.bar(
        mes_grav,
        x="MES_ES", y="Siniestros", color="GRAVEDAD",
        color_discrete_map=COLORES_GRAVEDAD,
        category_orders={"GRAVEDAD": ORDEN_GRAVEDAD, "MES_ES": meses_orden},
        labels={"MES_ES": "Mes"},
        title="Siniestros por mes del año",
    )
    fig_mes.update_layout(legend_title_text="Gravedad")
    aplicar_tema_fig(fig_mes)
    st.plotly_chart(fig_mes, use_container_width=True)
    st.caption(
        "Octubre, noviembre y diciembre muestran picos — fin de año con mayor tráfico y "
        "eventos nocturnos. Enero registra el mínimo anual."
    )

st.divider()

# ── Comparación año a año (líneas) ────────────────────────────────────────────
st.subheader("Evolución mensual por año")

mes_anio = (
    df.groupby(["ANIO", "MES"])
    .size()
    .reset_index(name="Siniestros")
)
mes_anio["MES_ES"] = mes_anio["MES"].map(MESES_ES)

fig_evol = px.line(
    mes_anio,
    x="MES_ES", y="Siniestros", color="ANIO",
    category_orders={"MES_ES": [MESES_ES[m] for m in range(1, 13)]},
    labels={"MES_ES": "Mes", "ANIO": "Año"},
    markers=True,
    title="Siniestros mensuales comparados por año (2015–2021)",
    color_discrete_sequence=px.colors.sequential.Greys[2:],
)
fig_evol.update_layout(legend_title_text="Año")

# Anotar caída COVID
anio_2020 = mes_anio[(mes_anio["ANIO"] == 2020) & (mes_anio["MES"] == 3)]
if not anio_2020.empty:
    fig_evol.add_annotation(
        x="Mar", y=anio_2020["Siniestros"].values[0],
        text="Inicio cuarentena<br>COVID-19",
        showarrow=True, arrowhead=2,
        ax=60, ay=-50,
    )

aplicar_tema_fig(fig_evol)
st.plotly_chart(fig_evol, use_container_width=True)
st.caption(
    "La caída brusca de 2020 a partir de marzo refleja las cuarentenas estrictas del COVID-19. "
    "En 2019 Bogotá registró su pico histórico con 32,962 siniestros."
)

st.divider()

# ── Mapa de calor día × mes ────────────────────────────────────────────────────
st.subheader("Mapa de calor: día de semana × mes")

heatmap_data = (
    df.groupby(["DIA_SEMANA_ES", "MES"])
    .size()
    .reset_index(name="Siniestros")
    .pivot(index="DIA_SEMANA_ES", columns="MES", values="Siniestros")
    .reindex(DIAS_ORDEN)
)
heatmap_data.columns = [MESES_ES[c] for c in heatmap_data.columns]

fig_heat = px.imshow(
    heatmap_data,
    color_continuous_scale="Greys",
    aspect="auto",
    labels=dict(color="Siniestros"),
    title="Concentración de siniestros por día y mes",
)
fig_heat.update_layout(
    xaxis_title="Mes",
    yaxis_title="Día de semana",
)
aplicar_tema_fig(fig_heat)
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ── Tabla: siniestros por localidad y año ─────────────────────────────────────
st.subheader("Tabla: siniestros por localidad y año")

tabla = (
    df.groupby(["LOCALIDAD", "ANIO"])
    .size()
    .unstack(fill_value=0)
)
tabla["Total"] = tabla.sum(axis=1)
# FIX: ordenar por "Total" (columna calculada), no por df["ANIO"].max()
# que puede no existir como columna si el filtro excluye ese año.
tabla = tabla.sort_values("Total", ascending=False)

st.dataframe(
    tabla.style.background_gradient(
        cmap="Greys",
        subset=[c for c in tabla.columns if c != "Total"]
    ).format("{:,}"),
    use_container_width=True,
    height=420,
)
