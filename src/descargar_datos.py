"""
Descarga de datos de siniestros viales de Bogotá desde el API de ArcGIS.

Fuente: Secretaría Distrital de Movilidad — Datos Abiertos Bogotá
API: https://services2.arcgis.com/NEwhEo9GGSHXcRXV/arcgis/rest/services/HistoricoSiniestros/FeatureServer/0

Estrategia: paginación de 10,000 registros sin geometría (outGeometry=false)
hasta descargar los ~210k registros históricos.
"""
import requests
import pandas as pd
import time
from pathlib import Path

URL_API = (
    "https://services2.arcgis.com/NEwhEo9GGSHXcRXV/arcgis/rest/services"
    "/HistoricoSiniestros/FeatureServer/0/query"
)

CAMPOS = ",".join([
    "OBJECTID",
    "CODIGO_ACCIDENTE",
    "FECHA_OCURRENCIA_ACC",
    "ANO_OCURRENCIA_ACC",
    "GRAVEDAD",
    "CLASE_ACC",
    "LOCALIDAD",
    "HORA",
    "LATITUD",
    "LONGITUD",
    "DIRECCION",
])

RUTA_SALIDA = Path(__file__).parent.parent / "data" / "siniestros.parquet"

TAMANO_PAGINA = 2_000  # límite real del FeatureServer


def descargar_pagina(offset: int) -> list[dict]:
    """Descarga una página de registros desde el API."""
    params = {
        "where": "1=1",
        "outFields": CAMPOS,
        "returnGeometry": "false",
        "resultOffset": offset,
        "resultRecordCount": TAMANO_PAGINA,
        "f": "json",
        "orderByFields": "OBJECTID ASC",
    }
    resp = requests.get(URL_API, params=params, timeout=30)
    resp.raise_for_status()
    datos = resp.json()
    if "features" not in datos:
        print(f"  Respuesta inesperada en offset {offset}: {datos.get('error', datos)}")
        return []
    return [f["attributes"] for f in datos["features"]]


def descargar_todo(ruta_salida: Path = RUTA_SALIDA) -> pd.DataFrame:
    """Descarga todos los registros paginando hasta agotar la fuente."""
    todos = []
    offset = 0
    pagina = 1

    print("Iniciando descarga de siniestros viales Bogotá...")
    while True:
        print(f"  Página {pagina} (offset={offset})...", end=" ", flush=True)
        registros = descargar_pagina(offset)
        if not registros:
            print("vacía — fin.")
            break
        todos.extend(registros)
        print(f"{len(registros)} registros acumulados={len(todos)}")
        if len(registros) < TAMANO_PAGINA:
            break  # última página
        offset += TAMANO_PAGINA
        pagina += 1
        time.sleep(0.3)  # respetar la API

    print(f"\nTotal descargado: {len(todos):,} registros")
    df = pd.DataFrame(todos)
    df = limpiar(df)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ruta_salida, index=False)
    print(f"Guardado en {ruta_salida}")
    return df


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y normaliza el DataFrame descargado."""
    # Convertir timestamps de epoch (ms) a datetime
    if "FECHA_OCURRENCIA_ACC" in df.columns:
        df["FECHA_OCURRENCIA_ACC"] = pd.to_datetime(
            df["FECHA_OCURRENCIA_ACC"], unit="ms", utc=True
        ).dt.tz_convert("America/Bogota").dt.tz_localize(None)

    # Columnas derivadas
    if "FECHA_OCURRENCIA_ACC" in df.columns:
        df["ANIO"] = df["ANO_OCURRENCIA_ACC"].fillna(
            df["FECHA_OCURRENCIA_ACC"].dt.year
        ).astype("Int64")
        df["MES"] = df["FECHA_OCURRENCIA_ACC"].dt.month
        df["DIA_SEMANA"] = df["FECHA_OCURRENCIA_ACC"].dt.day_name()
        df["DIA_SEMANA_ES"] = df["FECHA_OCURRENCIA_ACC"].dt.dayofweek.map({
            0: "Lunes", 1: "Martes", 2: "Miércoles",
            3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo",
        })

    # Hora como número
    if "HORA" in df.columns:
        df["HORA_NUM"] = pd.to_numeric(
            df["HORA"].str.split(":").str[0], errors="coerce"
        ).astype("Int64")

    # Normalizar textos
    for col in ["GRAVEDAD", "CLASE_ACC", "LOCALIDAD"]:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    # Filtrar coordenadas inválidas (0,0 o fuera de Bogotá)
    df = df[
        df["LATITUD"].between(4.0, 5.0) &
        df["LONGITUD"].between(-74.5, -73.9)
    ]

    return df.reset_index(drop=True)


if __name__ == "__main__":
    descargar_todo()
