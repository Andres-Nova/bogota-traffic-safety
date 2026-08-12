"""
Métricas de evaluación para el clasificador de gravedad de siniestros viales.

Métricas reportadas:
- auc_roc:   discriminación general (principal para comparar modelos)
- f1:        balance precisión/recall con umbral 0.5
- recall:    cobertura de accidentes graves (métrica operacional clave)
- pr_auc:    honesta con el desbalance de clases
- ks:        máxima separación entre distribuciones — define umbral operacional
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
        modelo:    Pipeline sklearn (preprocesador + clasificador).
        X_muestra: DataFrame con FEATURES_COLS (1 o más filas).

    Returns:
        (shap_values: np.ndarray, feature_names: list[str])
    """
    import shap  # lazy — solo cuando se necesita

    clasificador  = modelo.named_steps['modelo']
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
        nombres = list(preprocesador.get_feature_names_out())
    except Exception:
        nombres = [f'f{i}' for i in range(X_transformado.shape[1])]

    return shap_vals, nombres
