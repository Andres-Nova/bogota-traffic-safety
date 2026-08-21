# 🚦 Siniestralidad Vial — Bogotá D.C.

[![Demo LIVE](https://img.shields.io/badge/Demo-LIVE-brightgreen?style=flat-square&logo=streamlit)](https://bogota-traffic-safety.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-AUC%200.78-lightgrey?style=flat-square)](https://lightgbm.readthedocs.io)
[![Tests](https://img.shields.io/badge/Tests-10%20passing-brightgreen?style=flat-square)](tests/)

Construí este dashboard para analizar los patrones de accidentalidad vial en Bogotá usando los datos oficiales de la Secretaría Distrital de Movilidad, disponibles como datos abiertos. El resultado es una herramienta exploratoria que permite entender dónde, cuándo y con qué severidad ocurren los siniestros en la ciudad.

El dataset base es el [Histórico de Siniestros Viales](https://datosabiertos.bogota.gov.co/en/dataset/historico-siniestros-bogota-d-c) publicado por la SDM a través de su FeatureServer de ArcGIS. Cubre **209,861 siniestros** entre 2015 y 2021, con coordenadas geográficas, gravedad, tipo de accidente, localidad y fecha.

---

## Demo

**Dashboard interactivo:** [bogota-traffic-safety.streamlit.app](https://bogota-traffic-safety.streamlit.app/)

---

## Qué muestra el dashboard

### Página 1 — Resumen Ejecutivo
KPIs globales (total, fatales, heridos, solo daños), evolución anual con la caída del COVID-19 visible en 2020, distribución por gravedad y mapa de calor por localidad.

### Página 2 — Mapa de Siniestros
Mapa interactivo georreferenciado con filtros por año y gravedad. Incluye las intersecciones con más siniestros graves y la concentración por localidad.

### Página 3 — Análisis Temporal
Distribución por mes, día de semana y comparación año a año. El mapa de calor día × mes permite identificar los momentos de mayor riesgo a lo largo del año.

---

## Hallazgos principales

| Indicador | Valor |
|---|---|
| Total siniestros (2015–2021) | 209,861 |
| Con muertos | 3,393 (1.6 %) |
| Con heridos | 71,656 (34.1 %) |
| Solo daños | 134,812 (64.2 %) |
| Localidad más afectada | Kennedy (25,116) |
| Año con más siniestros | 2019 (32,962) |
| Año con menos siniestros | 2020 (22,424) — COVID-19 |
| Tipo más frecuente | Choque (85.8 %) |
| Día más peligroso | Jueves |
| Mes con más siniestros | Octubre–Diciembre |

### El impacto del COVID-19 en la accidentalidad

El año 2020 registró una **caída del 32 % respecto a 2019**, pasando de 32,962 a 22,424 siniestros. La reducción se concentra entre marzo y agosto, período de cuarentenas estrictas. Este patrón es estadísticamente significativo: no es ruido, es la huella del lockdown en la movilidad de la ciudad.

Desde el punto de vista de datos, esto hace que 2020 no sea directamente comparable con los demás años para análisis de tendencias — es un outlier estructural, no aleatorio.

### Patrón día de semana

El jueves concentra más siniestros que cualquier otro día (33,853), seguido del lunes (32,502). El sábado es el día más seguro (20,618). Esto es consistente con el patrón de movilidad bogotano: el tráfico laboral de mitad de semana es el más denso, mientras el sábado hay menos viajes de commuting.

### Patrón mensual y fin de año

Octubre, noviembre y diciembre concentran los picos anuales. El aumento coincide con el incremento de actividad económica, eventos nocturnos y mayor presencia de peatones en vías durante la temporada de fin de año. Enero registra el mínimo.

---

## Fuente de datos

- **Dataset:** Histórico Siniestros Bogotá D.C. — SDM / Datos Abiertos Bogotá
- **API:** `services2.arcgis.com/NEwhEo9GGSHXcRXV/arcgis/rest/services/HistoricoSiniestros/FeatureServer/0`
- **Período:** 2015–2021 | **Registros:** 209,861
- **Sistema de registro:** SIGAT (Sistema de Información Geográfica de Accidentes de Tránsito)
- **Licencia:** Creative Commons Attribution 4.0

La descarga se hace mediante paginación del FeatureServer (2,000 registros por request × 105 páginas). El resultado se almacena como `.parquet` para carga rápida en el dashboard.

---

## Estructura del proyecto

```
bogota-traffic-safety/
├── src/
│   └── descargar_datos.py      # Descarga paginada desde API ArcGIS → parquet
├── dashboard/
│   ├── app.py                  # Entrada multipage Streamlit
│   ├── utils.py                # Carga datos, filtros sidebar, constantes
│   └── pages/
│       ├── 1_resumen.py        # KPIs, tendencia anual, gravedad, localidades
│       ├── 2_mapa.py           # Mapa interactivo + top intersecciones
│       └── 3_analisis.py       # Mes, día semana, heatmap, tabla por localidad
├── data/
│   └── siniestros.parquet      # 209,861 registros limpios (7.7 MB)
├── tests/
│   └── test_datos.py           # 10 tests de integridad del dataset
└── .github/workflows/
    └── test.yml                # CI en cada push
```

---

## Correr localmente

```bash
git clone https://github.com/Andres-Nova/bogota-traffic-safety.git
cd bogota-traffic-safety

python -m venv .venv
source .venv/bin/activate
pip install -r dashboard/requirements.txt

streamlit run dashboard/app.py
```

Para re-descargar los datos desde la API:

```bash
python src/descargar_datos.py
```

## Tests

```bash
pytest tests/ -v
# 10 tests: integridad del parquet, coordenadas, gravedad, fechas, localidades
```

---

## ✍️ Autor

**Andres Nova** — AI Solutions Architect  
[andres-nova.github.io](https://andres-nova.github.io) · [LinkedIn](https://linkedin.com/in/andres-nova-data)
