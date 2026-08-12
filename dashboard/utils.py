"""Utilidades compartidas entre páginas del dashboard."""
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

RUTA_PARQUET  = Path(__file__).parent.parent / "data" / "siniestros.parquet"
RUTA_MODELO   = Path(__file__).parent.parent / "models" / "modelo_gravedad.pkl"
RUTA_KMEANS   = Path(__file__).parent.parent / "models" / "kmeans_zonas.pkl"

# Umbral operacional — prioriza recall de casos graves
UMBRAL_OPERACIONAL = 0.35

# Mapeo día de semana nombre → número (dayofweek: 0=lunes)
DIA_A_NUM = {
    "Lunes":     0,
    "Martes":    1,
    "Miércoles": 2,
    "Jueves":    3,
    "Viernes":   4,
    "Sábado":    5,
    "Domingo":   6,
}

# Paleta de colores por gravedad
COLORES_GRAVEDAD = {
    "Solo Danos":   "#3B82F6",   # azul
    "Con Heridos":  "#F59E0B",   # naranja
    "Con Muertos":  "#EF4444",   # rojo
}

# Orden fijo de gravedad (de menor a mayor)
ORDEN_GRAVEDAD = ["Solo Danos", "Con Heridos", "Con Muertos"]

# Días de semana en orden natural
DIAS_ORDEN = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

# Meses en español
MESES_ES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Ago",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}


@st.cache_data(show_spinner="Cargando datos...")
def cargar_datos() -> pd.DataFrame:
    """Carga el parquet de siniestros con caché de Streamlit."""
    df = pd.read_parquet(RUTA_PARQUET)
    return df


@st.cache_resource(show_spinner="Cargando modelo...")
def cargar_modelo():
    """
    Carga el Pipeline LightGBM y el KMeans de zonas desde disco.
    Usa cache_resource para que se carguen una sola vez por sesión.

    Returns:
        (pipeline, kmeans)
    """
    import joblib
    pipeline = joblib.load(RUTA_MODELO)
    kmeans   = joblib.load(RUTA_KMEANS)
    return pipeline, kmeans


def predecir_muestra(
    clase_acc: str,
    localidad: str,
    mes: int,
    dia_semana: str,
    es_corredor: bool,
    latitud: float,
    longitud: float,
) -> dict:
    """
    Construye features para una observación y predice la probabilidad de gravedad.

    Args:
        clase_acc:   Tipo de accidente (ej. 'Choque').
        localidad:   Nombre de la localidad de Bogotá.
        mes:         Mes numérico 1–12.
        dia_semana:  Nombre del día ('Lunes'…'Domingo').
        es_corredor: True si la dirección es una avenida principal.
        latitud:     Coordenada latitud decimal.
        longitud:    Coordenada longitud decimal.

    Returns:
        Dict con:
            probabilidad  (float 0–1)
            etiqueta      ('Alto' | 'Medio' | 'Bajo')
            color         (hex string)
            semaforo      (emoji)
    """
    pipeline, kmeans = cargar_modelo()

    # Encoding cíclico
    mes_sin = float(np.sin(2 * np.pi * mes / 12))
    mes_cos = float(np.cos(2 * np.pi * mes / 12))
    dia_num = DIA_A_NUM.get(dia_semana, 0)
    dia_sin = float(np.sin(2 * np.pi * dia_num / 7))
    dia_cos = float(np.cos(2 * np.pi * dia_num / 7))

    # Cluster de zona
    coords       = np.array([[latitud, longitud]])
    cluster_zona = int(kmeans.predict(coords)[0])

    fila = pd.DataFrame([{
        'CLASE_ACC':    clase_acc,
        'LOCALIDAD':    localidad,
        'MES_SIN':      mes_sin,
        'MES_COS':      mes_cos,
        'DIA_SIN':      dia_sin,
        'DIA_COS':      dia_cos,
        'LATITUD':      latitud,
        'LONGITUD':     longitud,
        'ES_CORREDOR':  int(es_corredor),
        'CLUSTER_ZONA': cluster_zona,
    }])

    probabilidad = float(pipeline.predict_proba(fila)[0, 1])

    if probabilidad >= 0.70:
        etiqueta = 'Alto'
        color    = '#EF4444'
        semaforo = '🔴'
    elif probabilidad >= UMBRAL_OPERACIONAL:
        etiqueta = 'Medio'
        color    = '#F59E0B'
        semaforo = '🟡'
    else:
        etiqueta = 'Bajo'
        color    = '#22C55E'
        semaforo = '🟢'

    return {
        'probabilidad': probabilidad,
        'etiqueta':     etiqueta,
        'color':        color,
        'semaforo':     semaforo,
    }


def sidebar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renderiza filtros en el sidebar y retorna el DataFrame filtrado.
    Filtros: Año y Gravedad.
    """
    st.sidebar.header("Filtros")

    anios = sorted(df["ANIO"].dropna().unique().tolist())
    anio_sel = st.sidebar.multiselect(
        "Año", anios, default=anios, key="filtro_anio"
    )

    gravedades = ORDEN_GRAVEDAD
    grav_sel = st.sidebar.multiselect(
        "Gravedad", gravedades, default=gravedades, key="filtro_gravedad"
    )

    localidades = sorted(df["LOCALIDAD"].dropna().unique().tolist())
    loc_sel = st.sidebar.multiselect(
        "Localidad (todas)", localidades, default=localidades, key="filtro_localidad"
    )

    mascara = (
        df["ANIO"].isin(anio_sel) &
        df["GRAVEDAD"].isin(grav_sel) &
        df["LOCALIDAD"].isin(loc_sel)
    )
    return df[mascara]
