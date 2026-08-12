# Spec: Clasificador de Gravedad de Siniestros Viales — Bogotá

**Fecha:** 2026-08-11
**Proyecto:** `bogota-traffic-safety`
**Repo:** https://github.com/Andres-Nova/bogota-traffic-safety

---

## Problema y objetivo

El dashboard actual de siniestros viales Bogotá muestra qué pasó (análisis histórico). Este spec agrega un modelo ML que predice **si un siniestro producirá víctimas**, dado el tipo de accidente, la ubicación, la localidad y el momento.

**Target:** `GRAVE` (binario)
- `1` = Con Heridos o Con Muertos (75,049 casos — 35.8 %)
- `0` = Solo Daños (134,812 casos — 64.2 %)

Utilidad práctica: priorización de respuesta de emergencias, identificación de corredores de alto riesgo, insumo para políticas viales.

---

## Dataset

- **Fuente:** API ArcGIS SDM — Histórico Siniestros Bogotá D.C.
- **Archivo:** `data/siniestros.parquet` (209,861 registros, 2015–2021, ya en repo)
- **Split:** 80 % train / 20 % test, estratificado por `GRAVE`
- **`random_state=42`** en todo

---

## Feature engineering (`src/features/build_features.py`)

| Feature | Tipo | Derivación | Justificación |
|---------|------|------------|---------------|
| `CLASE_ACC_ENC` | ordinal | OrdinalEncoder sobre `CLASE_ACC` (7 valores) | Predictor dominante: atropellos 99.8 % graves, choques 26 % |
| `LOCALIDAD_ENC` | ordinal | OrdinalEncoder sobre `LOCALIDAD` (~20 valores) | Zona de la ciudad — Caracas 55 % graves vs Suba 29 % |
| `MES_SIN`, `MES_COS` | cíclico | `sin/cos(2π × MES / 12)` | Estacionalidad sin ruptura entre dic y ene |
| `DIA_SIN`, `DIA_COS` | cíclico | `sin/cos(2π × DIA_SEM_NUM / 7)` | Semana laboral vs fin de semana sin ruptura dom–lun |
| `LATITUD`, `LONGITUD` | numérico | directo del parquet | Posición geográfica exacta |
| `ES_CORREDOR` | binario | `DIRECCION.startswith('AV ')` | Avenidas de alta velocidad concentran más muertes |
| `CLUSTER_ZONA` | ordinal | KMeans k=20 sobre (lat, lon), fit en train | Agrupa puntos calientes sin usar el target |

**Pipeline sklearn por modelo:**

```
ColumnTransformer(
  OrdinalEncoder → [CLASE_ACC, LOCALIDAD]
  StandardScaler → [LATITUD, LONGITUD, MES_SIN, MES_COS, DIA_SIN, DIA_COS]
  passthrough    → [ES_CORREDOR, CLUSTER_ZONA]   ← CLUSTER_ZONA ya es int de KMeans
) → clasificador
```

El `OrdinalEncoder` está dentro del pipeline para evitar data leakage en cross-validation. El `KMeans` (k=20) se ajusta **solo sobre el train set** y se serializa en `models/kmeans_zonas.pkl` — se carga al momento de transformar nuevas muestras en el dashboard.

---

## Modelos (`src/models/train.py`)

Se comparan 3 modelos con manejo explícito del desbalance:

| Modelo | Parámetro de desbalance | Rol |
|--------|------------------------|-----|
| Regresión Logística | `class_weight='balanced'` | Baseline interpretable |
| Random Forest | `class_weight='balanced'` | Ensemble robusto |
| **LightGBM** | `scale_pos_weight = 134812/75049 ≈ 1.80` | Candidato principal |

Ganador elegido por **ROC-AUC** en test. Pipeline completo (preprocesador + clasificador) serializado en `models/modelo_gravedad.pkl` y commiteado al repo (< 50 MB).

---

## Métricas (`src/models/evaluate.py`)

| Métrica | Función | Foco |
|---------|---------|------|
| ROC-AUC | `roc_auc_score` | Comparación entre modelos |
| F1 | `f1_score` | Balance precision/recall |
| **Recall** | `recall_score` | **Métrica operacional clave**: un accidente grave no detectado es más costoso que una falsa alarma |
| PR-AUC | `average_precision_score` | Honesta con el desbalance |
| Umbral óptimo | KS statistic | Punto donde se maximiza separación buenos/malos |

SHAP: `TreeExplainer` para LightGBM y RF, `LinearExplainer` para LR. Import lazy (igual que en `credit-risk-colombia`) — si SHAP no está disponible, el simulador omite la barra de factores sin crashear.

---

## Dashboard — Página 4: Predictor de Gravedad (`dashboard/pages/4_predictor.py`)

**Formulario de entrada (sidebar o columna izquierda):**
- `Tipo de accidente` — selectbox con los 7 valores de `CLASE_ACC`
- `Localidad` — selectbox con las ~20 localidades
- `Mes` — slider 1–12 con etiqueta del mes en español
- `Día de semana` — selectbox Lunes…Domingo
- `¿En avenida principal?` — checkbox

**Resultado (columna derecha):**
```
🔴 ACCIDENTE GRAVE  — probabilidad: 87 %
🟢 SOLO DAÑOS       — probabilidad: 13 %

Semáforo de riesgo:
  > 70 %  →  🔴 Alto
  40–70 % →  🟡 Medio
  < 40 %  →  🟢 Bajo

Top 3 factores (barra horizontal SHAP, si disponible):
  Atropello   ████████ +0.42
  Santa Fe    ████     +0.18
  Domingo     ██▌      -0.09   ← negativo = reduce riesgo
```

**Integración con Página 2 (Mapa):** al cargar el parquet, `utils.py` ejecuta `modelo.predict_proba()` en batch y agrega la columna `RIESGO_PREDICHO` (float 0–1). El tooltip de cada punto en el mapa muestra "Riesgo predicho: Alto/Medio/Bajo". La capa es opt-in via checkbox en sidebar para no degradar el rendimiento del mapa con 20 k+ puntos.

---

## Estructura de archivos (delta sobre estado actual)

```
bogota-traffic-safety/
├── src/
│   ├── descargar_datos.py          ← sin cambios
│   ├── features/
│   │   └── build_features.py       ← NUEVO
│   └── models/
│       ├── train.py                ← NUEVO
│       └── evaluate.py             ← NUEVO
├── dashboard/
│   ├── utils.py                    ← sin cambios
│   └── pages/
│       ├── 1_resumen.py … 3_analisis.py  ← sin cambios
│       └── 4_predictor.py          ← NUEVO
├── models/
│   ├── modelo_gravedad.pkl         ← NUEVO (commiteado, pipeline completo)
│   └── kmeans_zonas.pkl            ← NUEVO (commiteado, KMeans k=20 ajustado en train)
├── notebooks/
│   ├── 01_features.ipynb           ← NUEVO
│   └── 02_modelos.ipynb            ← NUEVO
├── data/
│   └── siniestros.parquet          ← sin cambios
└── tests/
    ├── test_datos.py               ← sin cambios (10 tests)
    ├── test_features.py            ← NUEVO (5 tests)
    └── test_modelo.py              ← NUEVO (5 tests)
```

---

## Tests nuevos

**`tests/test_features.py`** (5 tests):
1. `test_columnas_generadas` — todas las features esperadas existen en el output
2. `test_ciclicos_en_rango` — `MES_SIN`, `MES_COS`, `DIA_SIN`, `DIA_COS` ∈ [-1, 1]
3. `test_cluster_categorias_validas` — `CLUSTER_ZONA` ∈ {0, …, 19}
4. `test_corredor_binario` — `ES_CORREDOR` solo contiene 0 y 1
5. `test_sin_nulos_tras_features` — ninguna columna de features queda nula

**`tests/test_modelo.py`** (5 tests):
1. `test_modelo_carga` — `modelo_gravedad.pkl` existe y carga sin error
2. `test_predice_probabilidad` — `predict_proba` retorna valores ∈ [0, 1]
3. `test_forma_output` — output shape = (n_samples, 2)
4. `test_auc_minimo` — ROC-AUC en test ≥ 0.70 (piso de calidad)
5. `test_recall_minimo` — Recall clase 1 ≥ 0.60 (piso operacional)

---

## Criterios de éxito

| Métrica | Piso mínimo | Objetivo |
|---------|------------|---------|
| ROC-AUC | ≥ 0.70 | ≥ 0.75 |
| Recall (grave) | ≥ 0.60 | ≥ 0.70 |
| F1 | ≥ 0.55 | ≥ 0.65 |
| Tiempo de inferencia en dashboard | < 200 ms | < 100 ms |

---

## Convenciones

- Código y comentarios en español
- Sin Co-Authored-By ni metadatos de IA
- `random_state=42` en todo
- `models/modelo_gravedad.pkl` y `models/kmeans_zonas.pkl` commiteados (< 50 MB cada uno)
- SHAP: import lazy en `evaluate.py` — si no está disponible, page 4 muestra solo la probabilidad

---

## Fuera de alcance (este spec)

- Predicción de número de accidentes por zona+tiempo (serie temporal)
- Despliegue en infraestructura distinta a Streamlit Cloud
- Datos posteriores a 2021 (la API de la SDM no tiene más años disponibles aún)
- Re-entrenamiento automático programado
