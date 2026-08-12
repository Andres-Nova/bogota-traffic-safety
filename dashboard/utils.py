"""Utilidades compartidas entre páginas del dashboard."""
import pandas as pd
import streamlit as st
from pathlib import Path

RUTA_PARQUET = Path(__file__).parent.parent / "data" / "siniestros.parquet"

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
