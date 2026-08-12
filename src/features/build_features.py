"""
Feature engineering para el clasificador de gravedad de siniestros viales.

Entrada:  DataFrame del parquet (siniestros.parquet)
Salida:   DataFrame con FEATURES_COLS + instancia KMeans ajustada
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Columnas de features que consume el pipeline de entrenamiento
FEATURES_COLS = [
    'CLASE_ACC',    # categórico — será OrdinalEncoder en el pipeline
    'LOCALIDAD',    # categórico — será OrdinalEncoder en el pipeline
    'MES_SIN',      # cíclico seno del mes
    'MES_COS',      # cíclico coseno del mes
    'DIA_SIN',      # cíclico seno del día de semana
    'DIA_COS',      # cíclico coseno del día de semana
    'LATITUD',      # coordenada geográfica
    'LONGITUD',     # coordenada geográfica
    'ES_CORREDOR',  # binario: 1 si la dirección empieza con 'AV '
    'CLUSTER_ZONA', # entero 0-19: cluster KMeans de la posición
]

TARGET_COL = 'GRAVE'

# KMeans: número de clusters de zona
N_CLUSTERS = 20
RANDOM_STATE = 42


def construir_features(
    df: pd.DataFrame,
    kmeans: KMeans | None = None,
) -> tuple[pd.DataFrame, KMeans]:
    """
    Construye el DataFrame de features a partir del parquet de siniestros.

    Args:
        df:     DataFrame cargado desde siniestros.parquet.
        kmeans: KMeans ya ajustado (None = ajustar sobre df — usar solo en train).

    Returns:
        (X, kmeans) donde X tiene exactamente las columnas en FEATURES_COLS.
    """
    out = df.copy()

    # Target binario: 1 = con víctimas, 0 = solo daños
    out[TARGET_COL] = out['GRAVEDAD'].isin(['Con Heridos', 'Con Muertos']).astype(int)

    # Features cíclicas de mes (evita ruptura dic–ene)
    mes = out['MES'].astype(float)
    out['MES_SIN'] = np.sin(2 * np.pi * mes / 12)
    out['MES_COS'] = np.cos(2 * np.pi * mes / 12)

    # Features cíclicas de día de semana (0=lunes … 6=domingo)
    dia_num = out['FECHA_OCURRENCIA_ACC'].dt.dayofweek.astype(float)
    out['DIA_SIN'] = np.sin(2 * np.pi * dia_num / 7)
    out['DIA_COS'] = np.cos(2 * np.pi * dia_num / 7)

    # Indicador de corredor de alta velocidad
    out['ES_CORREDOR'] = out['DIRECCION'].str.upper().str.startswith('AV ').astype(int)

    # Cluster de zona geográfica — KMeans sobre (lat, lon)
    coords = out[['LATITUD', 'LONGITUD']].values
    if kmeans is None:
        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(coords)
    out['CLUSTER_ZONA'] = kmeans.predict(coords).astype(int)

    return out[FEATURES_COLS], kmeans
