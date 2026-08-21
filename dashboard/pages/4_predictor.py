"""
Página 4 — Predictor de Gravedad de Siniestros

El usuario describe un escenario (tipo de accidente, localidad, momento,
ubicación) y el modelo LightGBM predice la probabilidad de que resulte
con víctimas (heridos o muertos).  Se muestran:
  - Indicador semáforo con probabilidad
  - SHAP top-5 features (importancia para esa predicción)
  - Contexto histórico: tasa real para la combinación elegida
"""
import sys
from pathlib import Path

# Permite importar src.* desde el dashboard
_raiz = Path(__file__).parent.parent.parent
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from estilo import aplicar_estilo, toggle_tema_sidebar, aplicar_tema_fig

# ── Importa utilidades del dashboard ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    DIAS_ORDEN,
    DIA_A_NUM,
    MESES_ES,
    UMBRAL_OPERACIONAL,
    cargar_datos,
    cargar_modelo,
    predecir_muestra,
)

# ── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor · Siniestros Bogotá",
    page_icon="🚦",
    layout="wide",
)

toggle_tema_sidebar()
aplicar_estilo()
st.title("🚦 Predictor de Gravedad")
st.markdown(
    "Describe un escenario de accidente y el modelo estima la **probabilidad "
    "de que resulte con víctimas** (heridos o muertos)."
)
st.divider()

# ── Carga datos históricos (para listas de opciones y contexto) ────────────
df = cargar_datos()

clases_acc  = sorted(df["CLASE_ACC"].dropna().unique().tolist())
localidades = sorted(df["LOCALIDAD"].dropna().unique().tolist())

# ── Formulario de predicción ───────────────────────────────────────────────
col_form, col_resultado = st.columns([1, 1], gap="large")

with col_form:
    st.subheader("Parámetros del escenario")

    clase_acc = st.selectbox(
        "Tipo de accidente",
        clases_acc,
        index=clases_acc.index("Choque") if "Choque" in clases_acc else 0,
    )

    localidad = st.selectbox(
        "Localidad",
        localidades,
        index=localidades.index("KENNEDY") if "KENNEDY" in localidades else 0,
    )

    col_mes, col_dia = st.columns(2)
    with col_mes:
        mes = st.slider("Mes", min_value=1, max_value=12, value=6,
                        format="%d", help="1 = Enero … 12 = Diciembre")
        st.caption(f"▸ {MESES_ES.get(mes, mes)}")

    with col_dia:
        dia_semana = st.selectbox("Día de semana", DIAS_ORDEN, index=4)

    es_corredor = st.checkbox(
        "¿Ocurre en una avenida principal (AV …)?",
        value=False,
        help="Marcar si la vía es una avenida principal tipo AV CARACAS, AV BOYACÁ, etc.",
    )

    st.markdown("**Coordenadas (Bogotá)**")
    col_lat, col_lon = st.columns(2)
    with col_lat:
        latitud = st.number_input(
            "Latitud", value=4.6097, min_value=4.45, max_value=4.85,
            step=0.001, format="%.4f",
        )
    with col_lon:
        longitud = st.number_input(
            "Longitud", value=-74.0817, min_value=-74.25, max_value=-73.95,
            step=0.001, format="%.4f",
        )

    predecir = st.button("🔍 Predecir gravedad", type="primary", use_container_width=True)

# ── Resultado y SHAP ───────────────────────────────────────────────────────
with col_resultado:
    if predecir:
        with st.spinner("Calculando..."):
            resultado = predecir_muestra(
                clase_acc   = clase_acc,
                localidad   = localidad,
                mes         = mes,
                dia_semana  = dia_semana,
                es_corredor = es_corredor,
                latitud     = latitud,
                longitud    = longitud,
            )

        prob   = resultado["probabilidad"]
        etiq   = resultado["etiqueta"]
        color  = resultado["color"]
        sem    = resultado["semaforo"]

        # Tarjeta de resultado
        st.subheader("Resultado")
        st.markdown(
            f"""
            <div style="
                background: {color}22;
                border: 2px solid {color};
                border-radius: 12px;
                padding: 20px 24px;
                text-align: center;
            ">
                <div style="font-size: 3rem;">{sem}</div>
                <div style="font-size: 2rem; font-weight: 700; color: {color};">
                    {prob:.1%}
                </div>
                <div style="font-size: 1.1rem; color: {color}; font-weight: 600;">
                    Riesgo {etiq}
                </div>
                <div style="font-size: 0.85rem; color: #6B7280; margin-top: 8px;">
                    Umbral operacional: {UMBRAL_OPERACIONAL:.0%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── SHAP top-5 ────────────────────────────────────────────────────
        st.subheader("¿Por qué esta predicción?")
        try:
            from src.models.evaluate import calcular_shap_values

            pipeline, kmeans = cargar_modelo()

            # Reconstruye el DataFrame de features igual que en predecir_muestra
            mes_sin = float(np.sin(2 * np.pi * mes / 12))
            mes_cos = float(np.cos(2 * np.pi * mes / 12))
            dia_num = DIA_A_NUM.get(dia_semana, 0)
            dia_sin = float(np.sin(2 * np.pi * dia_num / 7))
            dia_cos = float(np.cos(2 * np.pi * dia_num / 7))
            coords  = np.array([[latitud, longitud]])
            cluster = int(kmeans.predict(coords)[0])

            fila_shap = pd.DataFrame([{
                'CLASE_ACC':    clase_acc,
                'LOCALIDAD':    localidad,
                'MES_SIN':      mes_sin,
                'MES_COS':      mes_cos,
                'DIA_SIN':      dia_sin,
                'DIA_COS':      dia_cos,
                'LATITUD':      latitud,
                'LONGITUD':     longitud,
                'ES_CORREDOR':  int(es_corredor),
                'CLUSTER_ZONA': cluster,
            }])

            shap_vals, nombres = calcular_shap_values(pipeline, fila_shap)
            shap_fila = shap_vals[0] if shap_vals.ndim == 2 else shap_vals

            # Simplifica nombres (quita prefijo ColumnTransformer)
            nombres_limpios = []
            for n in nombres:
                partes = n.split("__")
                nombres_limpios.append(partes[-1] if len(partes) > 1 else n)

            # Top-5 por valor absoluto
            idx_top = np.argsort(np.abs(shap_fila))[::-1][:5]
            top_nombres = [nombres_limpios[i] for i in idx_top]
            top_vals    = [float(shap_fila[i]) for i in idx_top]
            colores_bar = ['#EF4444' if v > 0 else '#3B82F6' for v in top_vals]

            fig_shap = go.Figure(go.Bar(
                x=top_vals[::-1],
                y=top_nombres[::-1],
                orientation='h',
                marker_color=colores_bar[::-1],
            ))
            fig_shap.update_layout(
                title="Top 5 factores (SHAP)",
                xaxis_title="Impacto en la predicción",
                yaxis_title="",
                height=300,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            aplicar_tema_fig(fig_shap)
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption("🔴 Aumenta el riesgo  ·  🔵 Reduce el riesgo")

        except ImportError:
            st.info("Instala `shap` para ver la explicación de la predicción.")
        except Exception as exc:
            st.warning(f"SHAP no disponible en este entorno: {exc}")

    else:
        st.subheader("Resultado")
        st.info("Configura los parámetros y pulsa **Predecir gravedad**.")

# ── Contexto histórico ─────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Contexto histórico")

if predecir:
    # Filtra registros con la combinación clase_acc + localidad
    df_ctx = df[
        (df["CLASE_ACC"] == clase_acc) &
        (df["LOCALIDAD"] == localidad)
    ].copy()

    if len(df_ctx) == 0:
        st.warning("No hay registros históricos para esta combinación.")
    else:
        df_ctx["GRAVE"] = df_ctx["GRAVEDAD"].isin(["Con Heridos", "Con Muertos"])
        tasa_historica  = df_ctx["GRAVE"].mean()
        total_acc       = len(df_ctx)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Accidentes registrados", f"{total_acc:,}")
        col_b.metric("Tasa grave histórica",    f"{tasa_historica:.1%}")
        col_c.metric("Predicción del modelo",
                     f"{resultado['probabilidad']:.1%}",
                     delta=f"{resultado['probabilidad'] - tasa_historica:+.1%}",
                     delta_color="inverse")

        st.caption(
            f"Combinación: **{clase_acc}** en **{localidad}** — "
            f"datos 2015–2024 · {total_acc:,} registros"
        )

        # ── Desglose temporal: días pico y horas pico ─────────────────────
        st.markdown("#### Distribución temporal en esta localidad")
        col_d, col_h = st.columns(2)

        with col_d:
            if "DIA_SEMANA_ES" in df_ctx.columns:
                conteo_dia = (
                    df_ctx["DIA_SEMANA_ES"]
                    .value_counts()
                    .reindex(DIAS_ORDEN, fill_value=0)
                )
                fig_dia = go.Figure(go.Bar(
                    x=conteo_dia.index.tolist(),
                    y=conteo_dia.values,
                    marker_color=["#e53e3e" if v == conteo_dia.max() else "#718096"
                                  for v in conteo_dia.values],
                ))
                fig_dia.update_layout(
                    title=f"Accidentes por día — {localidad}",
                    xaxis_title="Día", yaxis_title="Accidentes",
                    height=260, margin=dict(l=10, r=10, t=40, b=10),
                )
                aplicar_tema_fig(fig_dia)
                st.plotly_chart(fig_dia, use_container_width=True)
                dia_pico = conteo_dia.idxmax()
                st.caption(f"🔴 Día con más accidentes: **{dia_pico}**")

        with col_h:
            if "HORA_NUM" in df_ctx.columns:
                df_h = df_ctx["HORA_NUM"].dropna().astype(int)
                conteo_hora = df_h.value_counts().sort_index()
                fig_hora = go.Figure(go.Bar(
                    x=conteo_hora.index.tolist(),
                    y=conteo_hora.values,
                    marker_color=["#e53e3e" if v == conteo_hora.max() else "#718096"
                                  for v in conteo_hora.values],
                ))
                fig_hora.update_layout(
                    title=f"Accidentes por hora — {localidad}",
                    xaxis_title="Hora del día", yaxis_title="Accidentes",
                    height=260, margin=dict(l=10, r=10, t=40, b=10),
                )
                aplicar_tema_fig(fig_hora)
                st.plotly_chart(fig_hora, use_container_width=True)
                hora_pico = int(conteo_hora.idxmax())
                st.caption(f"🔴 Hora pico: **{hora_pico:02d}:00 – {hora_pico:02d}:59**")
else:
    st.info("El contexto histórico aparecerá tras predecir.")
