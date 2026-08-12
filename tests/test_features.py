"""Tests de feature engineering del clasificador de gravedad."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

RUTA_PARQUET = Path(__file__).parent.parent / "data" / "siniestros.parquet"

FEATURES_ESPERADAS = [
    'CLASE_ACC', 'LOCALIDAD',
    'MES_SIN', 'MES_COS', 'DIA_SIN', 'DIA_COS',
    'LATITUD', 'LONGITUD',
    'ES_CORREDOR', 'CLUSTER_ZONA',
]


@pytest.fixture(scope="module")
def df_muestra():
    """1000 filas del parquet — suficiente para tests rápidos."""
    return pd.read_parquet(RUTA_PARQUET).head(1000)


def test_columnas_generadas(df_muestra):
    """Todas las features esperadas deben existir en el output."""
    from src.features.build_features import construir_features
    X, _ = construir_features(df_muestra)
    faltantes = [c for c in FEATURES_ESPERADAS if c not in X.columns]
    assert not faltantes, f"Columnas faltantes: {faltantes}"


def test_ciclicos_en_rango(df_muestra):
    """MES_SIN, MES_COS, DIA_SIN, DIA_COS deben estar en [-1, 1]."""
    from src.features.build_features import construir_features
    X, _ = construir_features(df_muestra)
    for col in ['MES_SIN', 'MES_COS', 'DIA_SIN', 'DIA_COS']:
        assert X[col].between(-1, 1).all(), f"{col} fuera de [-1, 1]"


def test_cluster_categorias_validas(df_muestra):
    """CLUSTER_ZONA debe ser entero en {0, ..., 19}."""
    from src.features.build_features import construir_features
    X, _ = construir_features(df_muestra)
    assert X['CLUSTER_ZONA'].between(0, 19).all()
    assert X['CLUSTER_ZONA'].dtype in [np.int32, np.int64, int]


def test_corredor_binario(df_muestra):
    """ES_CORREDOR solo debe contener 0 y 1."""
    from src.features.build_features import construir_features
    X, _ = construir_features(df_muestra)
    valores = set(X['ES_CORREDOR'].unique())
    assert valores <= {0, 1}, f"Valores inesperados en ES_CORREDOR: {valores}"


def test_sin_nulos_tras_features(df_muestra):
    """Ninguna columna de features debe quedar nula."""
    from src.features.build_features import construir_features
    X, _ = construir_features(df_muestra)
    nulos = X[FEATURES_ESPERADAS].isnull().sum()
    cols_con_nulos = nulos[nulos > 0].index.tolist()
    assert not cols_con_nulos, f"Columnas con nulos: {cols_con_nulos}"
