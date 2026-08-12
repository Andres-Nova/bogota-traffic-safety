"""
Página 2 — Mapa de Siniestros

Mapa interactivo con puntos georreferenciados de siniestros viales en Bogotá.
Filtros por año y gravedad. Muestra top intersecciones más peligrosas.
Opción de capa de riesgo predicho por el modelo LightGBM.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from utils import (
    cargar_datos, sidebar_filtros,
    COLORES_GRAVEDAD, ORDEN_GRAVEDAD,
    cargar_modelo,
)

st.set_page_config(page_title="Mapa — Siniestralidad Bogotá", page_icon="🗺️", layout="wide")
st.title("🗺️ Mapa de Siniestros Viales")
st.caption("Bogotá D.C. · 2015–2021 · Fuente: SDM — Datos Abiertos Bogotá")

df_base = cargar_datos()
df = sidebar_filtros(df_base)

if df.empty:
    st.warning("Sin datos con los filtros seleccionados.")
    st.stop()

# ── Controles adicionales para el mapa ────────────────────────────────────────
st.sidebar.divider()
st.sidebar.subheader("Opciones del mapa")
max_puntos = st.sidebar.slider(
    "Máx. puntos en mapa",
    min_value=5_000, max_value=50_000, value=20_000, step=5_000,
    help="Más puntos = mapa más detallado pero más lento."
)
opacidad = st.sidebar.slider("Opacidad puntos", 0.1, 1.0, 0.4, step=0.1)
tamano = st.sidebar.slider("Tamaño puntos", 2, 10, 4)

st.sidebar.divider()
colorear_riesgo = st.sidebar.checkbox(
    "🔮 Colorear por riesgo predicho",
    value=False,
    help="Aplica el modelo LightGBM para estimar el riesgo de cada punto. "
         "Puede tardar unos segundos.",
)

# ── Muestrear si hay muchos puntos ────────────────────────────────────────────
if len(df) > max_puntos:
    df_mapa = df.sample(max_puntos, random_state=42)
    st.info(f"Mostrando {max_puntos:,} de {len(df):,} siniestros (muestra aleatoria).")
else:
    df_mapa = df.copy()

# ── Capa de riesgo predicho (opt-in) ──────────────────────────────────────────
COLORES_RIESGO  = {'Bajo': '#22C55E', 'Medio': '#F59E0B', 'Alto': '#EF4444'}
ORDEN_RIESGO    = ['Bajo', 'Medio', 'Alto']
UMBRAL_ALTO     = 0.70
UMBRAL_MEDIO    = 0.35

if colorear_riesgo:
    with st.spinner("Calculando riesgo predicho con el modelo LightGBM…"):
        try:
            from src.features.build_features import FEATURES_COLS, construir_features, TARGET_COL
            pipeline, kmeans = cargar_modelo()

            # Construir features para los puntos del mapa
            X_mapa, _ = construir_features(df_mapa.copy(), kmeans=kmeans)
            probas = pipeline.predict_proba(X_mapa)[:, 1]

            df_mapa = df_mapa.copy()
            df_mapa['RIESGO_PREDICHO'] = probas
            df_mapa['RIESGO_ETIQUETA'] = pd.cut(
                probas,
                bins=[-np.inf, UMBRAL_MEDIO, UMBRAL_ALTO, np.inf],
                labels=['Bajo', 'Medio', 'Alto'],
            ).astype(str)

            color_col   = 'RIESGO_ETIQUETA'
            color_map   = COLORES_RIESGO
            cat_orders  = {'RIESGO_ETIQUETA': ORDEN_RIESGO}
            hover_extra = {'RIESGO_PREDICHO': ':.1%'}
            leyenda_col = 'Riesgo predicho'
            titulo_mapa = f"Riesgo predicho — Bogotá ({len(df_mapa):,} puntos)"

        except Exception as exc:
            st.warning(f"No se pudo calcular el riesgo: {exc}")
            color_col   = 'GRAVEDAD'
            color_map   = COLORES_GRAVEDAD
            cat_orders  = {'GRAVEDAD': ORDEN_GRAVEDAD}
            hover_extra = {}
            leyenda_col = 'Gravedad'
            titulo_mapa = f"Siniestros viales Bogotá ({len(df_mapa):,} puntos)"
else:
    color_col   = 'GRAVEDAD'
    color_map   = COLORES_GRAVEDAD
    cat_orders  = {'GRAVEDAD': ORDEN_GRAVEDAD}
    hover_extra = {}
    leyenda_col = 'Gravedad'
    titulo_mapa = f"Siniestros viales Bogotá ({len(df_mapa):,} puntos)"

# ── Mapa principal ─────────────────────────────────────────────────────────────
hover_base = {
    "DIRECCION": True,
    "LOCALIDAD": True,
    "CLASE_ACC": True,
    "ANIO": True,
    "LATITUD": False,
    "LONGITUD": False,
}
hover_base.update(hover_extra)

fig_mapa = px.scatter_map(
    df_mapa,
    lat="LATITUD",
    lon="LONGITUD",
    color=color_col,
    color_discrete_map=color_map,
    category_orders=cat_orders,
    opacity=opacidad,
    size_max=tamano,
    zoom=10.5,
    center={"lat": 4.62, "lon": -74.08},
    hover_data=hover_base,
    labels={
        "GRAVEDAD": "Gravedad",
        "CLASE_ACC": "Tipo",
        "ANIO": "Año",
        "RIESGO_ETIQUETA":  leyenda_col,
        "RIESGO_PREDICHO":  "Probabilidad",
    },
    title=titulo_mapa,
    map_style="open-street-map",
    height=600,
)
fig_mapa.update_layout(
    legend=dict(
        orientation="v",
        yanchor="top", y=0.99,
        xanchor="left", x=0.01,
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="rgba(0,0,0,0.1)",
        borderwidth=1,
    ),
    margin=dict(l=0, r=0, t=40, b=0),
)
st.plotly_chart(fig_mapa, use_container_width=True)

# ── Top intersecciones más peligrosas ─────────────────────────────────────────
st.divider()
st.subheader("🔴 Intersecciones más peligrosas")
st.caption("Direcciones con más siniestros reportados (con muertos o heridos)")

df_graves = df[df["GRAVEDAD"].isin(["Con Muertos", "Con Heridos"])]
top_intersec = (
    df_graves.groupby("DIRECCION")
    .agg(
        total=("OBJECTID", "count"),
        muertos=("GRAVEDAD", lambda x: (x == "Con Muertos").sum()),
        heridos=("GRAVEDAD", lambda x: (x == "Con Heridos").sum()),
    )
    .sort_values("total", ascending=False)
    .head(15)
    .reset_index()
)
top_intersec.columns = ["Dirección", "Total siniestros", "Con muertos", "Con heridos"]
top_intersec.index = range(1, len(top_intersec) + 1)

# Colorear la fila si hay muertos
def colorear_fila(row):
    if row["Con muertos"] > 0:
        return ["background-color: #fee2e2"] * len(row)
    return [""] * len(row)

st.dataframe(
    top_intersec.style.apply(colorear_fila, axis=1),
    use_container_width=True,
    height=420,
)

# ── Mapa de calor por localidad ────────────────────────────────────────────────
st.divider()
st.subheader("Concentración por localidad")

loc_grav = (
    df.groupby(["LOCALIDAD", "GRAVEDAD"])
    .size()
    .reset_index(name="Siniestros")
)
loc_grav["GRAVEDAD"] = pd.Categorical(loc_grav["GRAVEDAD"], categories=ORDEN_GRAVEDAD, ordered=True)
loc_total = loc_grav.groupby("LOCALIDAD")["Siniestros"].sum().sort_values(ascending=False)

fig_loc = px.bar(
    loc_grav,
    x="LOCALIDAD",
    y="Siniestros",
    color="GRAVEDAD",
    color_discrete_map=COLORES_GRAVEDAD,
    category_orders={
        "GRAVEDAD": ORDEN_GRAVEDAD,
        "LOCALIDAD": loc_total.index.tolist(),
    },
    labels={"LOCALIDAD": "Localidad"},
    title="Siniestros por localidad y gravedad",
)
fig_loc.update_layout(
    xaxis_tickangle=-45,
    legend_title_text="Gravedad",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_loc, use_container_width=True)
