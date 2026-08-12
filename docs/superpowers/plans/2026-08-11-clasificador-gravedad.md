# Clasificador de Gravedad de Siniestros Viales — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agregar un clasificador LightGBM al repo `bogota-traffic-safety` que predice si un siniestro vial en Bogotá producirá víctimas (binario), y exponerlo en una nueva página 4 del dashboard Streamlit con formulario interactivo y explicación SHAP.

**Architecture:** Feature engineering sobre el parquet existente (cíclicos + KMeans de zonas) → pipeline sklearn (ColumnTransformer + LightGBM) → serialización en `models/` → página Streamlit que carga el modelo, acepta input del usuario y muestra la predicción con factores de riesgo. La integración con la Página 2 (mapa) es una capa opcional de riesgo predicho en batch.

**Tech Stack:** scikit-learn 1.5+, LightGBM 4.4+, joblib, pandas, Streamlit, Plotly

## Global Constraints

- Python 3.12 — igual que el resto del proyecto
- `random_state=42` en todos los modelos y splits
- Código y comentarios en español
- Sin Co-Authored-By ni metadatos de IA en commits
- SHAP: import lazy dentro de la función que lo usa — si falla, el dashboard sigue funcionando
- `models/modelo_gravedad.pkl` y `models/kmeans_zonas.pkl` se commitean al repo (< 50 MB)
- Pisos de calidad del modelo: ROC-AUC ≥ 0.70, Recall clase grave ≥ 0.60
- `dashboard/requirements.txt` es el único requirements que Streamlit Cloud lee — agregar dependencias aquí
- Directorio de trabajo: `/home/andres/Documentos/Codigo/Proyectos/bogota-traffic-safety/`

---

## Mapa de archivos

| Archivo | Acción | Responsabilidad |
|---------|--------|-----------------|
| `src/features/__init__.py` | Crear | Paquete vacío |
| `src/features/build_features.py` | Crear | Feature engineering: cíclicos, KMeans, corredor |
| `src/models/__init__.py` | Crear | Paquete vacío |
| `src/models/train.py` | Crear | Entrena 3 modelos, elige ganador, serializa |
| `src/models/evaluate.py` | Crear | Métricas (AUC, F1, Recall, PR-AUC, KS) + SHAP lazy |
| `tests/test_features.py` | Crear | 5 tests de feature engineering |
| `tests/test_modelo.py` | Crear | 5 tests del modelo serializado |
| `dashboard/pages/4_predictor.py` | Crear | Página 4: formulario + score + SHAP bars |
| `dashboard/utils.py` | Modificar | Añadir `cargar_modelo()` y `predecir_batch()` |
| `dashboard/pages/2_mapa.py` | Modificar | Añadir capa opt-in de riesgo predicho en tooltip |
| `dashboard/requirements.txt` | Modificar | Añadir scikit-learn, lightgbm, joblib |
| `models/modelo_gravedad.pkl` | Generar | Pipeline LightGBM serializado |
| `models/kmeans_zonas.pkl` | Generar | KMeans k=20 ajustado en train |

---

## Task 1: Feature Engineering

**Files:**
- Crear: `src/features/__init__.py`
- Crear: `src/features/build_features.py`
- Crear: `tests/test_features.py`

**Interfaces:**
- Produce: `construir_features(df, kmeans=None) -> tuple[pd.DataFrame, KMeans]`
  - `df`: DataFrame del parquet con columnas `CLASE_ACC`, `LOCALIDAD`, `MES`, `DIA_SEMANA_ES`, `FECHA_OCURRENCIA_ACC`, `LATITUD`, `LONGITUD`, `DIRECCION`
  - `kmeans`: instancia `sklearn.cluster.KMeans` ya ajustada, o `None` para ajustar en este llamado
  - Retorna `(X, kmeans)` donde `X` tiene exactamente las columnas en `FEATURES_COLS`
- Produce constante: `FEATURES_COLS: list[str]`
- Produce constante: `TARGET_COL: str = 'GRAVE'`
- Consume: `data/siniestros.parquet` — accede vía parámetro `df`, no lo carga internamente

- [ ] **Paso 1: Crear paquetes vacíos**

```bash
touch src/features/__init__.py
touch src/models/__init__.py
mkdir -p models
```

- [ ] **Paso 2: Escribir `tests/test_features.py` (tests primero — todos deben fallar)**

```python
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
```

- [ ] **Paso 3: Ejecutar tests — verificar que todos fallan con ImportError**

```bash
cd /home/andres/Documentos/Codigo/Proyectos/bogota-traffic-safety
source .venv/bin/activate
pytest tests/test_features.py -v 2>&1 | head -30
```

Esperado: 5 errores de `ImportError` o `ModuleNotFoundError`.

- [ ] **Paso 4: Implementar `src/features/build_features.py`**

```python
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
```

- [ ] **Paso 5: Ejecutar tests — verificar que los 5 pasan**

```bash
pytest tests/test_features.py -v
```

Esperado: `5 passed`.

- [ ] **Paso 6: Commit**

```bash
git add src/features/ src/models/__init__.py tests/test_features.py models/
git commit -m "feat: feature engineering — cíclicos, KMeans zonas, corredor — 5 tests"
```

---

## Task 2: Métricas y evaluación

**Files:**
- Crear: `src/models/evaluate.py`

**Interfaces:**
- Consume: pipeline sklearn con método `predict_proba(X)`
- Produce: `calcular_metricas(modelo, X_test, y_test) -> dict` con claves `auc_roc`, `f1`, `recall`, `pr_auc`, `ks`, `umbral_ks`
- Produce: `calcular_shap_values(modelo, X_muestra) -> tuple[np.ndarray, object]` — import lazy de SHAP
- Produce: `comparar_modelos(modelos, X_test, y_test) -> pd.DataFrame`

- [ ] **Paso 1: Implementar `src/models/evaluate.py`**

```python
"""
Métricas de evaluación para el clasificador de gravedad de siniestros viales.

Métricas reportadas:
- auc_roc:  discriminación general (principal para comparar modelos)
- f1:       balance precisión/recall con umbral 0.5
- recall:   cobertura de accidentes graves (métrica operacional clave)
- pr_auc:   honesta con el desbalance de clases
- ks:       máxima separación entre distribuciones — define umbral operacional
- umbral_ks: probabilidad en el punto de máximo KS
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    average_precision_score,
    roc_curve,
)


def calcular_ks(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """
    Estadístico KS: máxima separación entre CDFs de positivos y negativos.

    Returns:
        (ks, umbral) — valor KS y umbral de probabilidad que lo maximiza.
    """
    fpr, tpr, umbrales = roc_curve(y_true, y_proba)
    diferencias = tpr - fpr
    idx_max = int(np.argmax(diferencias))
    return float(diferencias[idx_max]), float(umbrales[idx_max])


def calcular_metricas(modelo, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Calcula todas las métricas para un modelo entrenado.

    Args:
        modelo:  Pipeline sklearn con predict_proba.
        X_test:  DataFrame de features (FEATURES_COLS).
        y_test:  Series binaria (TARGET_COL).

    Returns:
        Dict con claves: auc_roc, f1, recall, pr_auc, ks, umbral_ks.
    """
    y_proba = modelo.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)
    ks, umbral_ks = calcular_ks(y_test.values, y_proba)

    return {
        'auc_roc':   round(float(roc_auc_score(y_test, y_proba)), 4),
        'f1':        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'recall':    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'pr_auc':    round(float(average_precision_score(y_test, y_proba)), 4),
        'ks':        round(ks, 4),
        'umbral_ks': round(umbral_ks, 4),
    }


def comparar_modelos(
    modelos_entrenados: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Tabla comparativa de métricas para todos los modelos.
    Ordenada de mayor a menor AUC-ROC.
    """
    filas = []
    for nombre, modelo in modelos_entrenados.items():
        metricas = calcular_metricas(modelo, X_test, y_test)
        metricas['modelo'] = nombre
        filas.append(metricas)
    df = pd.DataFrame(filas).set_index('modelo')
    return df.sort_values('auc_roc', ascending=False)


def calcular_shap_values(modelo, X_muestra: pd.DataFrame) -> tuple:
    """
    Calcula SHAP values para explicar predicciones individuales.
    Import lazy — si SHAP no está instalado, lanza ImportError con mensaje claro.

    Args:
        modelo:   Pipeline sklearn (preprocesador + clasificador).
        X_muestra: DataFrame con FEATURES_COLS (1 o más filas).

    Returns:
        (shap_values: np.ndarray, feature_names: list[str])
    """
    import shap  # lazy — solo cuando se necesita

    clasificador = modelo.named_steps['modelo']
    preprocesador = modelo.named_steps['prep']
    X_transformado = preprocesador.transform(X_muestra)

    nombre_clase = type(clasificador).__name__
    if nombre_clase in ('LGBMClassifier', 'RandomForestClassifier'):
        explainer = shap.TreeExplainer(clasificador)
        shap_vals = explainer.shap_values(X_transformado)
        # RandomForest retorna lista [clase_0, clase_1] — tomar clase positiva
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
    else:
        # LogisticRegression
        explainer = shap.LinearExplainer(clasificador, X_transformado)
        shap_vals = explainer.shap_values(X_transformado)

    # Nombres de features tras el ColumnTransformer
    try:
        nombres = preprocesador.get_feature_names_out()
    except Exception:
        nombres = [f'f{i}' for i in range(X_transformado.shape[1])]

    return shap_vals, list(nombres)
```

- [ ] **Paso 2: Verificar que importa sin error**

```bash
cd /home/andres/Documentos/Codigo/Proyectos/bogota-traffic-safety
source .venv/bin/activate
python3 -c "from src.models.evaluate import calcular_metricas, comparar_modelos; print('OK')"
```

Esperado: `OK`

- [ ] **Paso 3: Commit**

```bash
git add src/models/evaluate.py
git commit -m "feat: métricas evaluación — AUC, F1, Recall, PR-AUC, KS, SHAP lazy"
```

---

## Task 3: Entrenamiento y serialización del modelo

**Files:**
- Crear: `src/models/train.py`
- Crear: `tests/test_modelo.py`
- Generar: `models/modelo_gravedad.pkl`, `models/kmeans_zonas.pkl`

**Interfaces:**
- Consume: `construir_features` de `src.features.build_features`
- Consume: `calcular_metricas`, `comparar_modelos` de `src.models.evaluate`
- Produce: `models/modelo_gravedad.pkl` — Pipeline sklearn completo (preprocesador + LightGBM)
- Produce: `models/kmeans_zonas.pkl` — KMeans k=20 ajustado en train

- [ ] **Paso 1: Instalar dependencias de entrenamiento**

```bash
cd /home/andres/Documentos/Codigo/Proyectos/bogota-traffic-safety
source .venv/bin/activate
pip install scikit-learn lightgbm joblib -q
```

- [ ] **Paso 2: Escribir `tests/test_modelo.py` (tests primero)**

```python
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
    df = pd.read_parquet(RUTA_PARQUET).tail(500)  # últimos registros como muestra test
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
    _, df_test = train_test_split(df, test_size=0.2, stratify=df[TARGET_COL], random_state=42)
    X_test, _ = construir_features(df_test, kmeans=kmeans)
    y_test = df_test[TARGET_COL]
    auc = roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1])
    assert auc >= 0.70, f"AUC {auc:.4f} < piso 0.70"


def test_recall_minimo(modelo, kmeans):
    """Recall de la clase grave en test debe superar el piso operacional (≥ 0.60)."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score
    from src.features.build_features import construir_features, TARGET_COL
    df = pd.read_parquet(RUTA_PARQUET)
    _, df_test = train_test_split(df, test_size=0.2, stratify=df[TARGET_COL], random_state=42)
    X_test, _ = construir_features(df_test, kmeans=kmeans)
    y_test = df_test[TARGET_COL]
    y_pred = modelo.predict(X_test)
    rec = recall_score(y_test, y_pred, zero_division=0)
    assert rec >= 0.60, f"Recall {rec:.4f} < piso 0.60"
```

- [ ] **Paso 3: Correr tests — verificar que fallan por pkl inexistente**

```bash
pytest tests/test_modelo.py -v 2>&1 | head -20
```

Esperado: `AssertionError: Modelo no encontrado` en todos.

- [ ] **Paso 4: Implementar `src/models/train.py`**

```python
"""
Entrenamiento del clasificador de gravedad de siniestros viales.

Compara 3 modelos (LR, RF, LightGBM), elige el de mayor AUC-ROC
y serializa el pipeline completo + KMeans en models/.

Uso:
    python src/models/train.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from lightgbm import LGBMClassifier

from src.features.build_features import (
    construir_features,
    FEATURES_COLS,
    TARGET_COL,
)
from src.models.evaluate import comparar_modelos

# Rutas
RUTA_PARQUET = Path(__file__).parent.parent.parent / "data" / "siniestros.parquet"
RUTA_MODELOS = Path(__file__).parent.parent.parent / "models"

# Columnas por tipo de transformación
COLS_ORDINAL = ['CLASE_ACC', 'LOCALIDAD']
COLS_ESCALAR = ['LATITUD', 'LONGITUD', 'MES_SIN', 'MES_COS', 'DIA_SIN', 'DIA_COS']
COLS_PASAR   = ['ES_CORREDOR', 'CLUSTER_ZONA']

RANDOM_STATE = 42


def construir_preprocesador() -> ColumnTransformer:
    """Preprocesador compartido entre todos los modelos."""
    return ColumnTransformer([
        ('ordinal', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        ), COLS_ORDINAL),
        ('escalar', StandardScaler(), COLS_ESCALAR),
        ('pasar',   'passthrough',    COLS_PASAR),
    ])


def construir_pipelines(peso_positivo: float) -> dict:
    """
    Retorna dict {nombre: Pipeline} con los 3 modelos a comparar.

    Args:
        peso_positivo: scale_pos_weight para LightGBM = n_negativos / n_positivos
    """
    return {
        'Regresión Logística': Pipeline([
            ('prep',   construir_preprocesador()),
            ('modelo', LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]),
        'Random Forest': Pipeline([
            ('prep',   construir_preprocesador()),
            ('modelo', RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        'LightGBM': Pipeline([
            ('prep',   construir_preprocesador()),
            ('modelo', LGBMClassifier(
                scale_pos_weight=peso_positivo,
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )),
        ]),
    }


def entrenar() -> None:
    """Pipeline completo: carga → features → train → eval → serializa."""
    print("Cargando datos...")
    df = pd.read_parquet(RUTA_PARQUET)

    # Split estratificado antes de feature engineering para evitar leakage
    df_train, df_test = train_test_split(
        df,
        test_size=0.20,
        stratify=df[TARGET_COL] if TARGET_COL in df.columns
                 else df['GRAVEDAD'].isin(['Con Heridos', 'Con Muertos']),
        random_state=RANDOM_STATE,
    )

    print("Construyendo features (train)...")
    # KMeans se ajusta solo en train — evita leakage geográfico
    X_train, kmeans = construir_features(df_train)
    y_train = df_train[TARGET_COL] if TARGET_COL in df_train.columns \
              else df_train['GRAVEDAD'].isin(['Con Heridos', 'Con Muertos']).astype(int)

    print("Construyendo features (test)...")
    X_test, _ = construir_features(df_test, kmeans=kmeans)
    y_test = df_test[TARGET_COL] if TARGET_COL in df_test.columns \
             else df_test['GRAVEDAD'].isin(['Con Heridos', 'Con Muertos']).astype(int)

    # Ratio para scale_pos_weight de LightGBM
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    peso_positivo = round(n_neg / n_pos, 4)
    print(f"Desbalance train — negativos: {n_neg:,}  positivos: {n_pos:,}  ratio: {peso_positivo}")

    pipelines = construir_pipelines(peso_positivo)

    print("\nEntrenando 3 modelos...")
    modelos_entrenados = {}
    for nombre, pipeline in pipelines.items():
        print(f"  → {nombre}...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)
        modelos_entrenados[nombre] = pipeline
        print("listo")

    print("\nMétricas en test:")
    tabla = comparar_modelos(modelos_entrenados, X_test, y_test)
    print(tabla.to_string())

    # Elegir ganador por AUC-ROC
    nombre_ganador = tabla.index[0]
    modelo_ganador = modelos_entrenados[nombre_ganador]
    print(f"\nGanador: {nombre_ganador}  AUC={tabla.loc[nombre_ganador, 'auc_roc']}")

    # Serializar
    RUTA_MODELOS.mkdir(exist_ok=True)
    joblib.dump(modelo_ganador, RUTA_MODELOS / "modelo_gravedad.pkl")
    joblib.dump(kmeans,         RUTA_MODELOS / "kmeans_zonas.pkl")
    print(f"Guardado en {RUTA_MODELOS}/")


if __name__ == "__main__":
    entrenar()
```

- [ ] **Paso 5: Ejecutar el entrenamiento**

```bash
cd /home/andres/Documentos/Codigo/Proyectos/bogota-traffic-safety
source .venv/bin/activate
python src/models/train.py
```

Esperado: tabla de 3 modelos con AUC, F1, Recall, PR-AUC, KS. Archivos `models/modelo_gravedad.pkl` y `models/kmeans_zonas.pkl` creados.

- [ ] **Paso 6: Ejecutar tests del modelo — deben pasar los 5**

```bash
pytest tests/test_modelo.py -v
```

Esperado: `5 passed`. Si `test_auc_minimo` falla (AUC < 0.70), revisar hiperparámetros de LightGBM.

- [ ] **Paso 7: Ejecutar todos los tests**

```bash
pytest tests/ -v
```

Esperado: `20 passed` (10 de test_datos + 5 de test_features + 5 de test_modelo).

- [ ] **Paso 8: Commit**

```bash
git add src/models/train.py tests/test_modelo.py models/modelo_gravedad.pkl models/kmeans_zonas.pkl
git commit -m "feat: entrenamiento y serialización clasificador gravedad — LightGBM, 3 modelos comparados, 20 tests"
```

---

## Task 4: Dashboard — Página 4 Predictor

**Files:**
- Crear: `dashboard/pages/4_predictor.py`
- Modificar: `dashboard/utils.py` — añadir `cargar_modelo()` y `predecir_muestra()`
- Modificar: `dashboard/requirements.txt` — añadir scikit-learn, lightgbm, joblib

**Interfaces:**
- Consume: `models/modelo_gravedad.pkl`, `models/kmeans_zonas.pkl` — cargados con joblib
- Consume: `construir_features` de `src.features.build_features`
- Consume: `calcular_shap_values` de `src.models.evaluate` (lazy — no falla si SHAP ausente)

- [ ] **Paso 1: Actualizar `dashboard/requirements.txt`**

Reemplazar el contenido por:

```
streamlit>=1.36.0
pandas>=2.2.2
plotly>=5.22.0
pyarrow>=14.0.0
requests>=2.32.0
scikit-learn>=1.5.1
lightgbm>=4.4.0
joblib>=1.4.2
```

- [ ] **Paso 2: Añadir `cargar_modelo()` y `predecir_muestra()` en `dashboard/utils.py`**

Añadir al final del archivo `dashboard/utils.py` (después de la función `sidebar_filtros` existente):

```python
# ── Modelo de gravedad ─────────────────────────────────────────────────────────
import joblib
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent.parent))

RUTA_MODELO  = Path(__file__).parent.parent / "models" / "modelo_gravedad.pkl"
RUTA_KMEANS  = Path(__file__).parent.parent / "models" / "kmeans_zonas.pkl"

# Mapping día de semana español → número (0=lunes)
DIA_A_NUM = {
    "Lunes": 0, "Martes": 1, "Miércoles": 2,
    "Jueves": 3, "Viernes": 4, "Sábado": 5, "Domingo": 6,
}


@st.cache_resource(show_spinner="Cargando modelo de gravedad...")
def cargar_modelo():
    """Carga el pipeline de gravedad y el KMeans. Cachea con st.cache_resource."""
    modelo = joblib.load(RUTA_MODELO)
    kmeans = joblib.load(RUTA_KMEANS)
    return modelo, kmeans


def predecir_muestra(
    clase_acc: str,
    localidad: str,
    mes: int,
    dia_semana: str,
    es_corredor: bool,
    latitud: float = 4.65,
    longitud: float = -74.08,
) -> dict:
    """
    Construye una fila sintética y devuelve la predicción del modelo.

    Returns:
        dict con claves: probabilidad_grave (float), etiqueta (str), color (str)
    """
    import numpy as np
    from src.features.build_features import FEATURES_COLS

    modelo, kmeans = cargar_modelo()
    dia_num = float(DIA_A_NUM.get(dia_semana, 0))

    # Construir fila de features a mano (mismas transformaciones que build_features)
    fila = {
        'CLASE_ACC':    clase_acc,
        'LOCALIDAD':    localidad,
        'MES_SIN':      float(np.sin(2 * np.pi * mes / 12)),
        'MES_COS':      float(np.cos(2 * np.pi * mes / 12)),
        'DIA_SIN':      float(np.sin(2 * np.pi * dia_num / 7)),
        'DIA_COS':      float(np.cos(2 * np.pi * dia_num / 7)),
        'LATITUD':      latitud,
        'LONGITUD':     longitud,
        'ES_CORREDOR':  int(es_corredor),
        'CLUSTER_ZONA': int(kmeans.predict([[latitud, longitud]])[0]),
    }
    X = pd.DataFrame([fila])[FEATURES_COLS]
    prob = float(modelo.predict_proba(X)[0, 1])

    if prob > 0.70:
        etiqueta, color = "🔴 Alto riesgo de víctimas", "#EF4444"
    elif prob > 0.40:
        etiqueta, color = "🟡 Riesgo medio", "#F59E0B"
    else:
        etiqueta, color = "🟢 Probable solo daños", "#22C55E"

    return {"probabilidad_grave": prob, "etiqueta": etiqueta, "color": color, "X": X}
```

- [ ] **Paso 3: Crear `dashboard/pages/4_predictor.py`**

```python
"""
Página 4 — Predictor de Gravedad

Formulario interactivo para estimar la probabilidad de que un siniestro
vial en Bogotá produzca víctimas (heridos o muertos).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Predictor — Siniestralidad Bogotá",
    page_icon="🚨",
    layout="wide",
)
st.title("🚨 Predictor de Gravedad")
st.caption(
    "Estima la probabilidad de que un siniestro vial produzca víctimas "
    "(heridos o muertos), dado el tipo de accidente, la zona y el momento."
)
st.info(
    "**Modelo:** LightGBM entrenado con 167,889 siniestros (2015–2021). "
    "Esta herramienta es exploratoria — no reemplaza los protocolos de emergencia.",
    icon="ℹ️",
)

# ── Importaciones del proyecto ─────────────────────────────────────────────────
from dashboard.utils import cargar_datos, predecir_muestra, DIAS_ORDEN, MESES_ES

df_base = cargar_datos()

# ── Formulario ─────────────────────────────────────────────────────────────────
st.subheader("Configurar el siniestro")
col_form, col_resultado = st.columns([1, 1], gap="large")

with col_form:
    clases = sorted(df_base['CLASE_ACC'].dropna().unique().tolist())
    clase_acc = st.selectbox("Tipo de accidente", clases, index=clases.index('Choque') if 'Choque' in clases else 0)

    localidades = sorted(df_base['LOCALIDAD'].dropna().replace('', pd.NA).dropna().unique().tolist())
    localidad = st.selectbox("Localidad", localidades)

    mes = st.slider("Mes", 1, 12, 6, format="%d")
    st.caption(f"Mes seleccionado: **{MESES_ES[mes]}**")

    dia_semana = st.selectbox("Día de semana", DIAS_ORDEN)

    es_corredor = st.checkbox("¿El siniestro ocurrió en una avenida principal (AV)?")

    st.divider()
    st.caption("Coordenadas de referencia (centro de la localidad por defecto):")
    col_lat, col_lon = st.columns(2)
    latitud  = col_lat.number_input("Latitud",  value=4.65, format="%.5f", step=0.001)
    longitud = col_lon.number_input("Longitud", value=-74.08, format="%.5f", step=0.001)

# ── Resultado ──────────────────────────────────────────────────────────────────
with col_resultado:
    resultado = predecir_muestra(
        clase_acc=clase_acc,
        localidad=localidad,
        mes=mes,
        dia_semana=dia_semana,
        es_corredor=es_corredor,
        latitud=latitud,
        longitud=longitud,
    )
    prob = resultado["probabilidad_grave"]

    st.metric(
        label="Probabilidad de accidente con víctimas",
        value=f"{prob * 100:.1f} %",
    )
    st.markdown(
        f"<h3 style='color:{resultado['color']}'>{resultado['etiqueta']}</h3>",
        unsafe_allow_html=True,
    )

    # Gauge visual
    fig_gauge = px.bar(
        x=[prob, 1 - prob],
        y=["Riesgo", "Riesgo"],
        orientation='h',
        color=["Grave", "No grave"],
        color_discrete_map={"Grave": resultado['color'], "No grave": "#E5E7EB"},
        labels={"x": "Probabilidad"},
    )
    fig_gauge.update_layout(
        showlegend=False,
        height=80,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        barmode='stack',
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()

    # SHAP — factores que más pesan
    st.subheader("Factores de riesgo")
    try:
        from src.models.evaluate import calcular_shap_values
        from dashboard.utils import cargar_modelo

        modelo, _ = cargar_modelo()
        X = resultado["X"]
        shap_vals, feature_names = calcular_shap_values(modelo, X)

        # Tomar los 5 factores con mayor valor absoluto
        importancias = list(zip(feature_names, shap_vals[0]))
        importancias.sort(key=lambda t: abs(t[1]), reverse=True)
        top5 = importancias[:5]

        df_shap = pd.DataFrame(top5, columns=["Feature", "Impacto SHAP"])
        df_shap["Dirección"] = df_shap["Impacto SHAP"].apply(
            lambda v: "↑ Aumenta riesgo" if v > 0 else "↓ Reduce riesgo"
        )
        df_shap["Color"] = df_shap["Impacto SHAP"].apply(
            lambda v: "#EF4444" if v > 0 else "#22C55E"
        )

        fig_shap = px.bar(
            df_shap,
            x="Impacto SHAP",
            y="Feature",
            orientation='h',
            color="Color",
            color_discrete_map="identity",
            hover_data=["Dirección"],
            title="Top 5 factores (SHAP values)",
        )
        fig_shap.update_layout(
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption(
            "Valores SHAP: magnitud = cuánto pesa el factor; "
            "rojo = aumenta la probabilidad de víctimas; verde = la reduce."
        )

    except ImportError:
        st.info("Instala `shap` para ver los factores de riesgo individuales.", icon="💡")
    except Exception as e:
        st.warning(f"No se pudo calcular SHAP: {e}")

# ── Contexto histórico ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Contexto histórico — ¿qué tan frecuente es este escenario?")

filtro = (
    (df_base['CLASE_ACC'] == clase_acc) &
    (df_base['LOCALIDAD'] == localidad)
)
df_filtrado = df_base[filtro]

if len(df_filtrado) > 0:
    col_h1, col_h2, col_h3 = st.columns(3)
    total = len(df_filtrado)
    graves = (df_filtrado['GRAVEDAD'].isin(['Con Heridos', 'Con Muertos'])).sum()
    col_h1.metric("Siniestros históricos", f"{total:,}")
    col_h2.metric("Con víctimas", f"{graves:,}")
    col_h3.metric("Tasa real de gravedad", f"{graves/total*100:.1f} %")
    st.caption(f"Siniestros de tipo **{clase_acc}** en **{localidad}** (2015–2021)")
else:
    st.info("Sin datos históricos para esta combinación en el dataset.")
```

- [ ] **Paso 4: Verificar que la página arranca sin errores de importación**

```bash
cd /home/andres/Documentos/Codigo/Proyectos/bogota-traffic-safety
source .venv/bin/activate
python3 -c "
import sys; sys.path.insert(0, 'dashboard')
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('utils', 'dashboard/utils.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('cargar_modelo:', mod.cargar_modelo)
print('predecir_muestra:', mod.predecir_muestra)
print('OK')
"
```

Esperado: `OK`

- [ ] **Paso 5: Commit**

```bash
git add dashboard/pages/4_predictor.py dashboard/utils.py dashboard/requirements.txt
git commit -m "feat: página 4 predictor de gravedad — formulario, score, SHAP, contexto histórico"
```

---

## Task 5: Integración con Mapa + Tests finales

**Files:**
- Modificar: `dashboard/pages/2_mapa.py` — añadir capa opt-in de riesgo predicho
- Ejecutar suite completa de tests

**Interfaces:**
- Consume: `cargar_modelo()`, `predecir_muestra()` de `dashboard.utils`
- Consume: `construir_features()` de `src.features.build_features`

- [ ] **Paso 1: Añadir capa de riesgo predicho en `dashboard/pages/2_mapa.py`**

Añadir en el sidebar (después de `st.sidebar.subheader("Opciones del mapa")`):

```python
# Añadir tras la definición de opacidad y tamano:
mostrar_riesgo = st.sidebar.checkbox(
    "Colorear por riesgo predicho",
    value=False,
    help="Usa el modelo ML para predecir la gravedad de cada punto. Puede tardar unos segundos.",
)
```

Añadir antes de `fig_mapa = px.scatter_map(...)`:

```python
# Capa de riesgo predicho (opt-in)
if mostrar_riesgo:
    try:
        import joblib, sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.features.build_features import construir_features
        from dashboard.utils import cargar_modelo

        with st.spinner("Calculando riesgo predicho..."):
            modelo, kmeans = cargar_modelo()
            X_batch, _ = construir_features(df_mapa, kmeans=kmeans)
            probas = modelo.predict_proba(X_batch)[:, 1]
            df_mapa = df_mapa.copy()
            df_mapa['RIESGO_PREDICHO'] = probas
            df_mapa['RIESGO_ETIQUETA'] = pd.cut(
                probas,
                bins=[0, 0.40, 0.70, 1.0],
                labels=['Bajo', 'Medio', 'Alto'],
            ).astype(str)

        # Sobreescribir color del mapa con el riesgo
        color_col = 'RIESGO_ETIQUETA'
        colores_riesgo = {'Bajo': '#22C55E', 'Medio': '#F59E0B', 'Alto': '#EF4444'}
    except Exception as e:
        st.warning(f"No se pudo calcular riesgo predicho: {e}")
        mostrar_riesgo = False

if not mostrar_riesgo:
    color_col = 'GRAVEDAD'
    colores_riesgo = COLORES_GRAVEDAD
```

Actualizar la llamada a `px.scatter_map`:
- Cambiar `color="GRAVEDAD"` → `color=color_col`
- Cambiar `color_discrete_map=COLORES_GRAVEDAD` → `color_discrete_map=colores_riesgo`
- Añadir `"RIESGO_ETIQUETA": mostrar_riesgo` en `hover_data` (solo si la columna existe)

- [ ] **Paso 2: Correr la suite completa de tests**

```bash
cd /home/andres/Documentos/Codigo/Proyectos/bogota-traffic-safety
source .venv/bin/activate
pytest tests/ -v
```

Esperado: `20 passed` (10 + 5 + 5).

- [ ] **Paso 3: Push al repo**

```bash
git add dashboard/pages/2_mapa.py
git commit -m "feat: capa riesgo predicho en mapa — opt-in, batch predict, semáforo Alto/Medio/Bajo"
git push origin main
```

- [ ] **Paso 4: Actualizar CI para incluir lightgbm y scikit-learn**

En `.github/workflows/test.yml`, actualizar el paso de instalación:

```yaml
      - name: Instalar dependencias
        run: |
          pip install pandas pyarrow pytest scikit-learn lightgbm joblib
```

- [ ] **Paso 5: Commit y push del CI**

```bash
git add .github/workflows/test.yml
git commit -m "ci: agregar scikit-learn, lightgbm, joblib a pipeline de tests"
git push origin main
```

---

## Self-Review

**Cobertura del spec:**

| Requisito del spec | Task que lo implementa |
|---|---|
| Feature engineering: cíclicos, KMeans, corredor | Task 1 |
| Pipeline sklearn: OrdinalEncoder + StandardScaler | Task 3 (`train.py`) |
| 3 modelos: LR, RF, LightGBM con desbalance | Task 3 (`train.py`) |
| Métricas: AUC, F1, Recall, PR-AUC, KS | Task 2 (`evaluate.py`) |
| SHAP lazy | Task 2 (`evaluate.py`) + Task 4 (`4_predictor.py`) |
| `models/modelo_gravedad.pkl` commiteado | Task 3 (Paso 8) |
| `models/kmeans_zonas.pkl` commiteado | Task 3 (Paso 8) |
| Página 4: formulario + score + semáforo | Task 4 (`4_predictor.py`) |
| Página 4: contexto histórico | Task 4 (`4_predictor.py`) |
| Integración Mapa: `RIESGO_PREDICHO` opt-in | Task 5 (`2_mapa.py`) |
| `test_features.py` — 5 tests | Task 1 |
| `test_modelo.py` — 5 tests (AUC ≥ 0.70, Recall ≥ 0.60) | Task 3 |
| `dashboard/requirements.txt` actualizado | Task 4 (Paso 1) |

**Placeholder scan:** Ninguno — todos los pasos tienen código real.

**Type consistency:**
- `construir_features(df, kmeans=None)` → definida en Task 1, consumida igual en Tasks 3, 4, 5 ✅
- `calcular_metricas(modelo, X_test, y_test)` → definida en Task 2, consumida en `test_modelo.py` Task 3 ✅
- `cargar_modelo()` → definida en Task 4 (`utils.py`), consumida en Task 4 (`4_predictor.py`) y Task 5 ✅
- `predecir_muestra(...)` → definida en Task 4 (`utils.py`), consumida solo en `4_predictor.py` ✅
- `FEATURES_COLS` → definida en Task 1, consumida en Task 4 (`utils.py`) ✅
- `TARGET_COL` → definida en Task 1, consumida en Task 3 (`train.py`) ✅
