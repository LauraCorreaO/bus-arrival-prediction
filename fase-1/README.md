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

## Selección y evaluación

La selección de algoritmo y de inclusión de `pasajeros` se realiza con una validación interna derivada únicamente del conjunto de entrenamiento. El test reservado no se consulta en los notebooks 03 y 04. El notebook 05 cierra la selección por MAE de validación, reajusta el candidato con todo el entrenamiento y calcula una sola evaluación final en test.

Se reportan MAE, RMSE y R² sobre las duraciones observadas. Las etiquetas positivas largas se conservan; una duración superior a la mediana de ruta más 12 horas no basta para confirmar un error. No se sustituyen objetivos por medianas. Se excluyen filas sin una salida o duración calculable, duraciones no positivas como errores confirmados y salidas fuera del período documentado (2023–2025), usando exclusivamente la fecha de salida para esa exclusión.

El patrón temporal de faltantes de `fecha_hora_salida_origen` (33,434 %) se informa como observado, pero no se interpreta por sí solo como evidencia de MCAR, MAR o MNAR. Se usa `fecha_salida` como fuente alternativa para derivar la salida cuando está disponible.

## Estado

- [x] Exploración de datos — `notebooks/01-exploracion-datos.ipynb`
- [x] Preparación de datos reproducible — `notebooks/02-preparacion-datos.ipynb`
- [x] Baseline, comparación y selección en validación — notebooks 03–05
- [x] Implementación de la evaluación final y exportación del pipeline

## Cómo ejecutar

```bash
# Python 3.13.x
pip install -r requirements.txt
jupyter notebook fase-1/notebooks/
```

Ejecutar los notebooks **en orden**: primero `01-exploracion-datos.ipynb` si se desea volver a descargar los datos, luego `02-preparacion-datos.ipynb` para generar los archivos de entrenamiento y prueba, después 03 y 04 para validación exploratoria, y por último 05 para selección y evaluación final.

### Datos incluidos en el repositorio

El repositorio incluye una copia de los datos, **comprimida** (`.csv.gz`) porque los CSV originales superan el límite de 100 MB de GitHub:

| Archivo | Contenido |
|---|---|
| `data/raw/datos_completos.csv.gz` | Datos crudos completos descargados de la API (2.617.130 filas) |
| `data/processed/train.csv.gz` | Conjunto de entrenamiento (80 %) |
| `data/processed/test.csv.gz` | Test reservado (20 %), usado solo en la evaluación final |

Los conjuntos procesados incluidos se pueden leer directamente para reproducir el análisis. Ejecuta 02 cada vez que regeneres la fuente o quieras incorporar cambios de preparación. Durante la selección solo se usa train; el acceso a test se deja al paso final del notebook 05.

```python
import pandas as pd

train = pd.read_csv("fase-1/data/processed/train.csv.gz")
```

Las predictoras son `nombre_sucursal`, `empresa`, `ruta_origen`, `subregi_n`, `clase_veh_culo`, `pasajeros`, `hora_salida_decimal`, `dia_semana` y `mes`; el objetivo observado es `duracion_viaje_horas`, en horas. `duracion_original` conserva una copia auditable de la etiqueta y nunca es predictora. Las fechas y horas de llegada se usan únicamente para calcular el objetivo y se excluyen de las entradas del modelo. `hora_salida_decimal`, `dia_semana` y `mes` se derivan de la fecha/hora de salida.

Todos los notebooks leen y escriben los conjuntos procesados en formato `.csv.gz`. Las dependencias de `requirements.txt` están fijadas a versiones exactas. Si se vuelve a ejecutar la preparación con una fuente distinta, cambian los conjuntos y se debe repetir el análisis de validación antes de la evaluación final.

Los modelos `.joblib` que ya estaban en `models/` pertenecen a la ejecución anterior y no se regeneraron en esta corrección. No deben considerarse el modelo producido por el protocolo actualizado: al ejecutar el notebook 05 se genera `models/modelo_final.joblib`.

## Estructura

```
data/raw/        Datos descargados de la API (versionados comprimidos, .csv.gz)
data/processed/  Conjuntos de entrenamiento y prueba (versionados comprimidos, .csv.gz)
notebooks/       Notebooks ejecutables
models/          Modelo entrenado
```
