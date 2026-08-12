"""
Entrenamiento y serialización del clasificador de gravedad de siniestros viales.

Modelos entrenados:
- LightGBM (principal, clase-balanceado con scale_pos_weight)
- Random Forest (comparación, árbol robusto)
- Regresión Logística (línea base lineal)

Salidas:
- models/modelo_gravedad.pkl   → Pipeline LightGBM (mejor AUC)
- models/kmeans_zonas.pkl      → KMeans ajustado solo en train
"""
import sys
from pathlib import Path

# Asegura que src/ sea importable al correr como script
_raiz = Path(__file__).parent.parent.parent
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from lightgbm import LGBMClassifier

from src.features.build_features import (
    FEATURES_COLS,
    TARGET_COL,
    construir_features,
)
from src.models.evaluate import comparar_modelos

# ── Rutas ──────────────────────────────────────────────────────────────────
RUTA_PARQUET  = _raiz / "data" / "siniestros.parquet"
RUTA_MODELO   = _raiz / "models" / "modelo_gravedad.pkl"
RUTA_KMEANS   = _raiz / "models" / "kmeans_zonas.pkl"

# ── Columnas por tipo de preprocesado ──────────────────────────────────────
COLS_ORDINAL  = ['CLASE_ACC', 'LOCALIDAD']
COLS_ESCALAR  = ['LATITUD', 'LONGITUD', 'MES_SIN', 'MES_COS', 'DIA_SIN', 'DIA_COS']
COLS_PASAR    = ['ES_CORREDOR', 'CLUSTER_ZONA']

RANDOM_STATE  = 42


def construir_preprocesador() -> ColumnTransformer:
    """ColumnTransformer compartido por todos los pipelines."""
    return ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1,
            ), COLS_ORDINAL),
            ('escalar', StandardScaler(), COLS_ESCALAR),
            ('pasar', 'passthrough', COLS_PASAR),
        ],
        remainder='drop',
    )


def construir_pipelines(escala_pos: float) -> dict:
    """
    Crea los tres pipelines con preprocesador propio cada uno.

    Args:
        escala_pos: n_negativos / n_positivos — corrige desbalance en LightGBM.
    """
    pipelines = {
        'LightGBM': Pipeline([
            ('prep', construir_preprocesador()),
            ('modelo', LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                scale_pos_weight=escala_pos,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )),
        ]),
        'RandomForest': Pipeline([
            ('prep', construir_preprocesador()),
            ('modelo', RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        'LogisticRegression': Pipeline([
            ('prep', construir_preprocesador()),
            ('modelo', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs',
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }
    return pipelines


def entrenar(ruta_parquet: Path = RUTA_PARQUET) -> dict:
    """
    Pipeline completo de entrenamiento:
    1. Carga datos
    2. Split estratificado 80/20
    3. Feature engineering (KMeans fit solo en train)
    4. Entrena tres modelos
    5. Compara métricas
    6. Serializa el mejor (LightGBM esperado) y el KMeans

    Returns:
        Dict con claves 'metricas' (DataFrame) y 'nombre_ganador' (str).
    """
    print("▶ Cargando datos...")
    df = pd.read_parquet(ruta_parquet)
    print(f"  {len(df):,} registros cargados")

    # Target binario en el df completo para estratificar el split
    df[TARGET_COL] = df['GRAVEDAD'].isin(['Con Heridos', 'Con Muertos']).astype(int)
    print(f"  Tasa de graves: {df[TARGET_COL].mean():.1%}")

    # Split estratificado
    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    print(f"  Train: {len(df_train):,} | Test: {len(df_test):,}")

    # Feature engineering — KMeans se ajusta SOLO en train
    print("▶ Construyendo features...")
    X_train, kmeans = construir_features(df_train)
    X_test, _       = construir_features(df_test, kmeans=kmeans)
    y_train = df_train[TARGET_COL]
    y_test  = df_test[TARGET_COL]

    # Peso para corregir desbalance de clases
    n_neg      = int((y_train == 0).sum())
    n_pos      = int((y_train == 1).sum())
    escala_pos = n_neg / n_pos
    print(f"  scale_pos_weight = {escala_pos:.3f} ({n_neg} neg / {n_pos} pos)")

    # Entrenamiento
    print("▶ Entrenando modelos...")
    pipelines     = construir_pipelines(escala_pos)
    entrenados    = {}
    for nombre, pipeline in pipelines.items():
        print(f"  [{nombre}]...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)
        entrenados[nombre] = pipeline
        print("listo")

    # Comparación de métricas
    print("▶ Evaluando métricas...")
    tabla = comparar_modelos(entrenados, X_test, y_test)
    print(tabla.to_string())

    # El ganador es el de mayor AUC-ROC
    nombre_ganador = tabla.index[0]
    modelo_ganador = entrenados[nombre_ganador]
    print(f"\n✓ Ganador: {nombre_ganador}  AUC={tabla.loc[nombre_ganador, 'auc_roc']:.4f}")

    # Serialización
    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo_ganador, RUTA_MODELO)
    joblib.dump(kmeans, RUTA_KMEANS)
    print(f"✓ Modelo guardado en: {RUTA_MODELO}")
    print(f"✓ KMeans guardado en: {RUTA_KMEANS}")

    return {
        'metricas':       tabla,
        'nombre_ganador': nombre_ganador,
    }


if __name__ == '__main__':
    resultado = entrenar()
    print(f"\nEntrenamiento completo. Modelo: {resultado['nombre_ganador']}")
