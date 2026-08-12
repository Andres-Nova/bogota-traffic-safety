"""Tests unitarios para el pipeline de datos de siniestros viales Bogotá."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

RUTA_PARQUET = Path(__file__).parent.parent / "data" / "siniestros.parquet"


@pytest.fixture(scope="module")
def df():
    """Carga el dataset una sola vez para todos los tests."""
    assert RUTA_PARQUET.exists(), f"Parquet no encontrado: {RUTA_PARQUET}"
    return pd.read_parquet(RUTA_PARQUET)


def test_parquet_existe():
    """El archivo de datos debe existir."""
    assert RUTA_PARQUET.exists()


def test_volumen_minimo(df):
    """El dataset debe tener al menos 200k registros."""
    assert len(df) >= 200_000, f"Solo {len(df):,} registros — se esperan ≥200k"


def test_columnas_requeridas(df):
    """Todas las columnas clave deben estar presentes."""
    cols_requeridas = [
        "OBJECTID", "GRAVEDAD", "CLASE_ACC", "LOCALIDAD",
        "LATITUD", "LONGITUD", "ANIO", "MES", "DIA_SEMANA_ES",
    ]
    faltantes = [c for c in cols_requeridas if c not in df.columns]
    assert not faltantes, f"Columnas faltantes: {faltantes}"


def test_coordenadas_bogota(df):
    """Todas las coordenadas deben estar dentro del área de Bogotá."""
    assert df["LATITUD"].between(4.0, 5.0).all(), "Hay latitudes fuera de Bogotá"
    assert df["LONGITUD"].between(-74.5, -73.9).all(), "Hay longitudes fuera de Bogotá"


def test_sin_coordenadas_nulas(df):
    """No debe haber coordenadas nulas tras la limpieza."""
    assert df["LATITUD"].notnull().all()
    assert df["LONGITUD"].notnull().all()


def test_gravedad_valores_validos(df):
    """GRAVEDAD solo debe tener los tres valores esperados."""
    valores_esperados = {"Solo Danos", "Con Heridos", "Con Muertos"}
    valores_reales = set(df["GRAVEDAD"].unique())
    inesperados = valores_reales - valores_esperados
    assert not inesperados, f"Valores inesperados en GRAVEDAD: {inesperados}"


def test_anios_cubren_periodo(df):
    """El dataset debe cubrir los años 2015–2021."""
    anios = set(df["ANIO"].dropna().unique())
    for anio in range(2015, 2022):
        assert anio in anios, f"Año {anio} falta en el dataset"


def test_localidades_no_nulas(df):
    """
    No debe haber localidades nulas.
    Toleramos < 0.1% de strings vacíos (46 registros de la SDM sin asignar).
    """
    assert df["LOCALIDAD"].notnull().all()
    pct_vacios = (df["LOCALIDAD"] == "").mean()
    assert pct_vacios < 0.001, f"Demasiados registros sin localidad: {pct_vacios:.2%}"


def test_mes_rango(df):
    """MES debe estar entre 1 y 12."""
    assert df["MES"].between(1, 12).all()


def test_dias_semana_validos(df):
    """DIA_SEMANA_ES solo debe tener los 7 días esperados."""
    dias_esperados = {"Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"}
    dias_reales = set(df["DIA_SEMANA_ES"].unique())
    inesperados = dias_reales - dias_esperados
    assert not inesperados, f"Días inesperados: {inesperados}"
