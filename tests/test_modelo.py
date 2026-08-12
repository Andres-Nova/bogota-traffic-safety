"""Tests del modelo de gravedad serializado."""
import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

RUTA_PARQUET = Path(__file__).parent.parent / "data" / "siniestros.parquet"
RUTA_MODELO  = Path(__file__).parent.parent / "models" / "modelo_gravedad.pkl"
RUTA_KMEANS  = Path(__file__).parent.parent / "models" / "kmeans_zonas.pkl"


@pytest.fixture(scope="module")
def modelo():
    assert RUTA_MODELO.exists(), f"Modelo no encontrado: {RUTA_MODELO}"
    return joblib.load(RUTA_MODELO)


@pytest.fixture(scope="module")
def kmeans():
    assert RUTA_KMEANS.exists(), f"KMeans no encontrado: {RUTA_KMEANS}"
    return joblib.load(RUTA_KMEANS)


@pytest.fixture(scope="module")
def X_test_muestra(kmeans):
    from src.features.build_features import construir_features
    df = pd.read_parquet(RUTA_PARQUET).tail(500)
    X, _ = construir_features(df, kmeans=kmeans)
    return X


def test_modelo_carga():
    """El archivo pkl existe y carga sin error."""
    modelo = joblib.load(RUTA_MODELO)
    assert modelo is not None


def test_predice_probabilidad(modelo, X_test_muestra):
    """predict_proba retorna valores entre 0 y 1."""
    probas = modelo.predict_proba(X_test_muestra)[:, 1]
    assert probas.min() >= 0.0
    assert probas.max() <= 1.0


def test_forma_output(modelo, X_test_muestra):
    """Output shape = (n_samples, 2)."""
    out = modelo.predict_proba(X_test_muestra)
    assert out.shape == (len(X_test_muestra), 2)


def test_auc_minimo(modelo, kmeans):
    """ROC-AUC en test debe superar el piso de calidad (≥ 0.70)."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from src.features.build_features import construir_features, TARGET_COL
    df = pd.read_parquet(RUTA_PARQUET)
    df[TARGET_COL] = df['GRAVEDAD'].isin(['Con Heridos', 'Con Muertos']).astype(int)
    _, df_test = train_test_split(df, test_size=0.2, stratify=df[TARGET_COL], random_state=42)
    X_test, _ = construir_features(df_test, kmeans=kmeans)
    y_test = df_test[TARGET_COL]
    auc = roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1])
    assert auc >= 0.70, f"AUC {auc:.4f} < piso 0.70"


def test_recall_minimo(modelo, kmeans):
    """
    Recall de la clase grave con umbral operacional 0.35.

    Para seguridad vial se prioriza sensibilidad (cobertura de casos graves)
    sobre especificidad: un falso negativo (accidente grave no detectado) tiene
    mayor costo operacional que un falso positivo.  El umbral 0.35 es el punto
    de corte elegido en producción — el test verifica que el modelo lo cumple.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score
    from src.features.build_features import construir_features, TARGET_COL
    UMBRAL_OPERACIONAL = 0.35
    df = pd.read_parquet(RUTA_PARQUET)
    df[TARGET_COL] = df['GRAVEDAD'].isin(['Con Heridos', 'Con Muertos']).astype(int)
    _, df_test = train_test_split(df, test_size=0.2, stratify=df[TARGET_COL], random_state=42)
    X_test, _ = construir_features(df_test, kmeans=kmeans)
    y_test = df_test[TARGET_COL]
    y_pred = (modelo.predict_proba(X_test)[:, 1] >= UMBRAL_OPERACIONAL).astype(int)
    rec = recall_score(y_test, y_pred, zero_division=0)
    assert rec >= 0.60, f"Recall con umbral {UMBRAL_OPERACIONAL} = {rec:.4f} < piso 0.60"
