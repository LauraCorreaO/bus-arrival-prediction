# Fase 1 — Predicción de la duración de un viaje en la Terminal de Transporte

## Integrantes
- Laura Correa Ochoa
- Miguel Cerquera Arias
- Santiago David López

## Descripción del problema

Predecir la **duración de un viaje** (en horas) entre su salida y su llegada a la Terminal de Transporte, usando únicamente información conocida al momento de la salida: ruta de origen, empresa, terminal, clase de vehículo, número de pasajeros, y hora, día de la semana y mes de salida. Es un problema de **regresión** (no series de tiempo): cada viaje es una observación independiente.

## Fuente de los datos

[Datos Abiertos Colombia — Llegadas y Salidas de Vehículos de la Terminal de Transporte](https://www.datos.gov.co/Transporte/Llegada-y-Salidas-de-Veh-culos-de-la-Terminal-de-T/pfsr-mdyi/about_data) (recurso `pfsr-mdyi`). Se descarga completo desde la API SoQL: 2.617.130 registros de 2023 a 2025, todas las clases de vehículo.

## Objetivo del modelo

Estimar el tiempo de viaje (ETA) de un vehículo que acaba de salir hacia la terminal, antes de que llegue.

## Algoritmo utilizado

Se compararon cuatro algoritmos, en dos líneas de trabajo independientes:

- `03-modelo-predictivo.ipynb`: `DummyRegressor` (mediana, modelo base), `Ridge` y `HistGradientBoostingRegressor`.
- `04-modelado-lineal-y-random-forest.ipynb`: regresión lineal (modelo base de esa línea) y `RandomForestRegressor`, cada uno con y sin la variable `pasajeros`.

Los dos mejores de cada línea (HistGradientBoosting y Random Forest sin `pasajeros`) se enfrentaron entre sí, bajo las mismas condiciones, en `05-comparacion-final-modelos.ipynb`.

**Modelo oficial: `RandomForestRegressor`** (sin `pasajeros`; 80 árboles, `min_samples_leaf=50`, `max_samples=0.5`), guardado en `models/modelo_random_forest.joblib`, por tener el MAE más bajo de los cuatro candidatos.

`HistGradientBoostingRegressor` (`models/modelo_hist_gradient_boosting.joblib`) queda documentado como alternativa: su precisión es prácticamente la misma (RMSE y R² casi empatados) y el archivo es ~175 veces más liviano (0,27 MB frente a 47 MB) y algo más rápido al predecir, una ventaja a tener en cuenta para el despliegue en las fases siguientes.

## Métrica empleada

**MAE** (error absoluto medio, en horas) es la métrica principal: para estimar un tiempo de llegada es la más natural de interpretar ("el modelo se equivoca en promedio X horas") y los pocos viajes extremadamente largos la afectan menos que al RMSE. Se reportan también **RMSE** y **R²** como métricas secundarias. Las tres se calculan además solo sobre las filas con duración real (`duracion_corregida == False`), para confirmar que el resultado no depende de los valores que se reemplazaron en la preparación (notebook 02).

## Principales resultados

Sobre el conjunto de prueba (523.425 filas):

| Modelo | MAE (h) | RMSE (h) | R² |
|---|---|---|---|
| Baseline (mediana) | 2,16 | 3,89 | -0,17 |
| Regresión lineal | 0,43 | 0,86 | 0,94 |
| HistGradientBoosting | 0,39 | 0,79 | 0,952 |
| **Random Forest (oficial)** | **0,36** | 0,79 | 0,952 |

El modelo oficial reduce el MAE del baseline en un 83 % (de 2,16 h a 0,36 h, unos 22 minutos de error promedio) y explica el 95 % de la variación de la duración del viaje.

## Estado

- [x] Exploración de datos — `notebooks/01-exploracion-datos.ipynb`
- [x] Preparación de datos — `notebooks/02-preparacion-datos.ipynb`
- [x] Modelo base y modelo predictivo — `notebooks/03-modelo-predictivo.ipynb`, `notebooks/04-modelado-lineal-y-random-forest.ipynb`, `notebooks/05-comparacion-final-modelos.ipynb`
- [x] Guardado del modelo — `models/modelo_random_forest.joblib` (oficial) y `models/modelo_hist_gradient_boosting.joblib` (alternativa documentada)

## Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter notebook fase-1/notebooks/
```

Ejecutar los notebooks **en orden y de principio a fin**:

1. `01-exploracion-datos.ipynb` — descarga los datos completos a `data/raw/` (tarda unos minutos).
2. `02-preparacion-datos.ipynb` — genera `train.csv` y `test.csv` en `data/processed/`.
3. `03-modelo-predictivo.ipynb` y `04-modelado-lineal-y-random-forest.ipynb` — entrenan y comparan los candidatos de cada línea de trabajo, y guardan cada uno su mejor modelo en `models/`. Se pueden ejecutar en cualquier orden entre sí, uno depende del otro solo a través de `train.csv`/`test.csv`.
4. `05-comparacion-final-modelos.ipynb` — carga los dos modelos ya entrenados (no reentrena nada) y elige el modelo oficial con la misma métrica y el mismo conjunto de prueba.

### Datos incluidos en el repositorio

El repositorio incluye una copia de los datos, **comprimida** (`.csv.gz`) porque los CSV originales superan el límite de 100 MB de GitHub:

| Archivo | Contenido |
|---|---|
| `data/raw/datos_completos.csv.gz` | Datos crudos completos descargados de la API (2.617.130 filas) |
| `data/processed/train.csv.gz` | Conjunto de entrenamiento (80 %) |
| `data/processed/test.csv.gz` | Conjunto de prueba (20 %) |

Para modelar no hace falta ejecutar los notebooks 01 y 02: pandas lee los archivos comprimidos directamente.

```python
import pandas as pd

train = pd.read_csv("fase-1/data/processed/train.csv.gz")
test = pd.read_csv("fase-1/data/processed/test.csv.gz")
```

Las predictoras son `nombre_sucursal`, `empresa`, `ruta_origen`, `subregi_n`, `clase_veh_culo`, `pasajeros`, `hora_salida_decimal`, `dia_semana` y `mes`; el objetivo es `duracion_viaje_horas`. La columna `duracion_corregida` **no es una predictora** (se deriva del objetivo): solo sirve para evaluar el modelo sobre las filas no corregidas.

Los `.csv` sin comprimir no se versionan; se generan al ejecutar los notebooks. Si se vuelven a ejecutar y cambian los datos, hay que volver a comprimirlos y subirlos.

## Estructura

```
data/raw/        Datos descargados de la API (versionados comprimidos, .csv.gz)
data/processed/  Conjuntos de entrenamiento y prueba (versionados comprimidos, .csv.gz)
notebooks/       Notebooks ejecutables
models/          Modelos entrenados
```

### Modelos guardados

`models/` contiene dos modelos, cada uno el resultado de una línea de trabajo distinta:

| Archivo | Estado |
|---|---|
| `modelo_random_forest.joblib` | **Oficial** — el que se usa en las siguientes fases del proyecto |
| `modelo_hist_gradient_boosting.joblib` | Alternativa documentada |
