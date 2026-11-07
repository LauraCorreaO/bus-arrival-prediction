# Fase 2: scripts y Docker para predecir la duración de un viaje en la Terminal de Transporte

## Integrantes
- Laura Correa Ochoa
- Miguel Cerquera Arias
- Santiago David López

## Problema y relación con la Fase 1

Se predice la **duración de un viaje en horas** (`duracion_viaje_horas`), desde su salida hasta su llegada a la Terminal de Transporte de Medellín. Solo se usa información conocida al momento de la salida: terminal, empresa, ruta de origen, subregión, clase de vehículo, pasajeros, y hora, día de la semana y mes de salida. Es un problema de **regresión**.

Se conservan el problema, el conjunto de datos y el enfoque de la [Fase 1](../fase-1/README.md): baseline de la mediana, regresión lineal y Random Forest. El trabajo de los notebooks se pasó a una aplicación reutilizable:

- `train.py` entrena y guarda una versión del modelo.
- `predict.py` carga una versión guardada y predice sobre datos nuevos.
- `src/` contiene la lógica compartida: datos, *features*, pipeline, evaluación y versionamiento.

## Correcciones respecto a la Fase 1

| Observación | Corrección en la Fase 2 |
|---|---|
| Los notebooks 03 a 05 usaban el test para elegir el algoritmo y decidir si incluir `pasajeros` (sesgo de selección). | `src/evaluation.py` (`select_model`) elige con una **validación interna derivada solo del conjunto de entrenamiento**. Luego reajusta con todo el entrenamiento y evalúa **una sola vez** en test, junto al modelo base y con la misma partición. |
| En 34 registros, el mes y el día se reconstruían con `hora_de_llegada`, que es información futura. | `src/data.py` deriva hora, día de la semana y mes **solo de la salida** (`fecha_hora_salida_origen` y, si falta, `fecha_salida`). La llegada solo se usa para calcular el objetivo. `src/config.py` declara las columnas de llegada en `LEAKAGE_COLUMNS`, y una prueba comprueba que ninguna predictora deriva de la llegada. |
| Las duraciones se reemplazaban por la mediana de la ruta, y una duración mayor que la mediana + 12 h se trataba como error sin confirmación. | **Se conserva la duración original.** Solo se excluyen errores confirmados: filas sin salida (5), salida fuera de 2023–2025 (45), sin duración calculable (5) y duración no positiva (209), en total 253 de 2.617.130. Una fila puede cumplir varios motivos. Los retrasos largos se mantienen. El conteo queda en `metadata.json` (`datos.exclusiones`). |
| El patrón de faltantes por período se presentaba como demostración de MAR. | Ya no se afirma MAR. |
| El notebook 02 escribía CSV, pero los notebooks 03 a 05 leían copias `.csv.gz` que no se actualizaban. | `train.py` lee directamente el archivo crudo (`.csv` o `.csv.gz`) y hace toda la preparación en una sola ejecución, sin archivos intermedios. Las versiones de los paquetes quedan fijadas en `requirements.txt`. |
| La importancia por permutación usaba R² por defecto y se presentaba como RMSE. | La Fase 2 no calcula importancia por permutación. Las métricas se nombran explícitamente: MAE (principal), RMSE y R², en horas. |
| Los PR apuntaban a `main`, sin pasar por `develop` y sin revisión. | El trabajo se integra con ramas `feature/*`, PR hacia `develop` revisados por otro integrante y, al final, `develop` → `main`. |

Además, el preprocesamiento (normalización de categorías, imputación y codificación) está dentro de un `Pipeline` de scikit-learn y se guarda con el modelo. Así se ajusta solo con el conjunto de entrenamiento y se aplica igual en la predicción.

## Entorno e instalación

- **Python 3.13.5**: es la versión registrada en `models/*/metadata.json`, y la imagen Docker usa la misma.
- Dependencias con versiones fijas en [`requirements.txt`](requirements.txt): pandas, numpy, scikit-learn, joblib y pytest.

Todos los comandos se ejecutan **desde la carpeta `fase-2`**.

PowerShell:
```powershell
cd fase-2
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Bash:
```bash
cd fase-2
python3.13 -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt
```

## Datos

**Fuente:** [Datos Abiertos Colombia: Llegadas y Salidas de Vehículos de la Terminal de Transporte](https://www.datos.gov.co/Transporte/Llegada-y-Salidas-de-Veh-culos-de-la-Terminal-de-T/pfsr-mdyi) (recurso `pfsr-mdyi`). Tiene 2.617.130 registros de 2023 a 2025 y no requiere credenciales.

**Ubicación esperada:** el archivo crudo completo ya está en el repositorio como `../fase-1/data/raw/datos_completos.csv.gz` (versionado en la Fase 1), así que no hace falta descargarlo. `train.py` lo lee directamente con `--data`. Si prefieres tenerlo dentro de `fase-2`, cópialo a `data/datos_completos.csv.gz`. Esa ruta está en `.gitignore` y no se sube.

El CSV crudo debe contener estas columnas:
- `nombre_sucursal`
- `empresa`
- `ruta_origen`
- `subregi_n`
- `clase_veh_culo`
- `pasajeros`
- `fecha_hora_salida_origen`
- `fecha_salida`
- `hora_de_llegada`

### Esquema del CSV de predicción

[`data/ejemplo_prediccion.csv`](data/ejemplo_prediccion.csv) tiene 6 filas reales del dataset, **sin la variable objetivo**. La última fila incluye celdas nulas a propósito.

| Columna | Tipo | Descripción |
|---|---|---|
| `nombre_sucursal` | texto | Terminal (`Terminal del Norte` o `Terminal del Sur`). No distingue mayúsculas. |
| `empresa` | texto | Empresa transportadora |
| `ruta_origen` | texto | Municipio de origen; el sufijo `-MEDELLIN` se elimina |
| `subregi_n` | texto | Subregión de Antioquia |
| `clase_veh_culo` | texto | `BUS`, `MICROBUS`, `AUTOMOVIL`, etc. |
| `pasajeros` | número ≥ 0 | Número de pasajeros (el modelo entregado no lo usa, pero la columna es obligatoria) |
| `hora_salida_decimal` | número en [0, 24) | Hora de salida, por ejemplo 14.5 = 14:30 |
| `dia_semana` | entero 0–6 | 0 = lunes … 6 = domingo |
| `mes` | entero 1–12 | Mes de salida |

**Reglas de la entrada** (implementadas en `predict.py`):
- El orden de las columnas no importa: se reordenan según el esquema guardado en `metadata.json` (`variables_entrada`).
- Una **celda nula** es válida. Las categóricas se tratan como `DESCONOCIDA` y las numéricas se imputan con la mediana aprendida en el entrenamiento.
- Una **columna ausente** produce un error y código de salida 2.
- Un **valor no numérico, no entero o fuera de rango** produce un error que indica las filas afectadas, con código de salida 8.
- Las **columnas adicionales** se descartan. Si viene la variable objetivo, se muestra una advertencia y también se descarta.
- Las **categorías no vistas** en el entrenamiento se aceptan. El Random Forest las codifica como `-1` y la regresión lineal las ignora en el one-hot.
- Ante cualquier error **no se escribe el CSV de salida**, porque se escribe primero en un temporal y luego se renombra.

## Entrenamiento: `train.py`

```bash
python train.py --help
```

Comandos con los que se generaron las versiones entregadas (en Bash o PowerShell, en una sola línea):

```bash
python train.py --data ../fase-1/data/raw/datos_completos.csv.gz --target duracion_viaje_horas --output-dir models --version v1 --seed 42
python train.py --data ../fase-1/data/raw/datos_completos.csv.gz --target duracion_viaje_horas --output-dir models --version v2 --seed 42 --n-estimators 40 --min-samples-leaf 100 --max-samples 0.3
```

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--data` | (obligatorio) | CSV crudo (`.csv` o `.csv.gz`) |
| `--target` | `duracion_viaje_horas` | Única variable objetivo admitida: se calcula como llegada − salida |
| `--output-dir` | `models` | Carpeta de versiones |
| `--version` | (obligatorio) | Identificador único. Si ya existe, el entrenamiento se rechaza (código 1) y no se sobrescribe |
| `--seed` | 42 | Semilla de la partición y del Random Forest |
| `--test-size` / `--val-size` | 0.2 / 0.2 | Proporción de test y de validación interna (dentro del entrenamiento) |
| `--n-estimators`, `--min-samples-leaf`, `--max-samples` | 80, 50, 0.5 | Hiperparámetros del Random Forest |
| `--max-rows` | todas | Usa una muestra para pruebas rápidas (mínimo 100) |

**Proceso:**
1. Lee y valida las columnas.
2. Deriva las predictoras de la salida.
3. Hace la partición aleatoria 80/20 antes de ajustar cualquier transformación.
4. Selecciona el modelo con validación interna: baseline, lineal y Random Forest, con y sin `pasajeros`. `pasajeros` solo se incluye si mejora la MAE al menos un 1 %.
5. Reajusta con todo el entrenamiento y evalúa una sola vez en test, junto al modelo base.
6. Guarda la versión.

El entrenamiento completo tarda unos 11 minutos (medido en Docker).

## Predicción: `predict.py`

```bash
python predict.py --help
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
python predict.py --model-version v2 --input data/ejemplo_prediccion.csv --output outputs/predicciones_v2.csv
```

| Argumento | Descripción |
|---|---|
| `--model` | Ruta al `modelo.joblib` o a la carpeta de la versión. Tiene prioridad sobre `--model-version`. |
| `--model-version` | Versión dentro de `--models-dir` (por defecto `v1`) |
| `--models-dir` | Carpeta de versiones (por defecto `models/`, junto a `predict.py`) |
| `--input` / `--output` | CSV de entrada y de salida (obligatorios) |

La salida contiene las predictoras en el orden del entrenamiento y la columna **`prediccion_duracion_horas`** (en horas), con **una fila por cada fila de entrada y en el mismo orden**. `predict.py` no reentrena ni reajusta: solo carga el pipeline guardado.

**Códigos de salida:**

| Código | Situación |
|---|---|
| 0 | Éxito |
| 1 | No existe el modelo o el metadata |
| 2 | Falta una columna |
| 4 | No existe el archivo de entrada |
| 5 | CSV ilegible |
| 6 | CSV vacío |
| 7 | Fallo al predecir |
| 8 | Tipo o rango inválido |

## Versiones, métricas y metadatos

```
models/
  v1/  modelo.joblib  metrics.json  metadata.json   (Random Forest: 80 árboles, min_samples_leaf 50, max_samples 0.5)
  v2/  modelo.joblib  metrics.json  metadata.json   (Random Forest: 40 árboles, min_samples_leaf 100, max_samples 0.3)
outputs/                                            (predicciones; se crea al ejecutar y no se versiona)
```

- **`metrics.json`** guarda las métricas en test del modelo y del baseline, la reducción de la MAE y las métricas de validación de cada candidato.
- **`metadata.json`** guarda:
  - versión, fecha, tipo de problema, variable objetivo y algoritmo;
  - variables de entrada con sus tipos y el tratamiento de nulos, categorías no vistas y columnas extra;
  - hiperparámetros y semilla;
  - datos usados: fuente, archivo, SHA-256, exclusiones y filas;
  - estrategia de separación y tamaños;
  - versión de Python y de los paquetes;
  - hash del commit.

Ambas versiones se cargan y predicen (ver las pruebas y la evidencia más abajo).

## Docker

Probado con Docker 29.3.1 en **PowerShell**, desde `fase-2`.

**Imagen:**
- Base `python:3.13.5-slim`, directorio de trabajo `/app`.
- No tiene `ENTRYPOINT`; el comando se indica al ejecutar.
- No entrena durante el build.
- Los datos, modelos y salidas se montan desde el equipo anfitrión, así que no se copian a la imagen.

**PowerShell:**
```powershell
docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t modelos-fase2:1.0 .
New-Item -ItemType Directory -Force models, outputs
docker run --rm -v "${PWD}\..\fase-1\data\raw:/app/data/raw:ro" -v "${PWD}\models:/app/models" modelos-fase2:1.0 python train.py --data data/raw/datos_completos.csv.gz --target duracion_viaje_horas --output-dir models --version v-docker --seed 42
docker run --rm -v "${PWD}\data:/app/data:ro" -v "${PWD}\models:/app/models:ro" -v "${PWD}\outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

**Bash** (se antepone `MSYS_NO_PATHCONV=1` a cada `docker run`):
```bash
docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t modelos-fase2:1.0 .
mkdir -p models outputs
docker run --rm -v "$PWD/../fase-1/data/raw:/app/data/raw:ro" -v "$PWD/models:/app/models" modelos-fase2:1.0 python train.py --data data/raw/datos_completos.csv.gz --target duracion_viaje_horas --output-dir models --version v-docker --seed 42
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models:ro" -v "$PWD/outputs:/app/outputs" modelos-fase2:1.0 python predict.py --model models/v-docker/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

`models/v-docker/` es una versión de verificación: persiste en el equipo anfitrión, pero está en `.gitignore` y no forma parte de la entrega.

### Evidencia de ejecución en Docker (2026-11-05)

- **Build:** sin errores. La imagen ocupa 657 MB en disco.
- **`train.py` en el contenedor:** terminó con código 0 en 638 s.

  ```
  Filas utilizables: 2,616,877 (excluidas: 253)
  Entrenamiento: 2,093,501 | prueba: 523,376
  Modelo seleccionado: Random Forest (sin pasajeros)
  Modelo   (Random Forest (sin pasajeros)): MAE 0.4322 h | RMSE 3.4886 h | R² 0.5118
  Baseline (mediana): MAE 2.2104 h | RMSE 5.2292 h | R² -0.0968
  ```

  Son las mismas métricas de `v1`, entrenada localmente, porque se usan los mismos datos, la misma semilla y el mismo entorno.
- **`predict.py` en el contenedor:** `Predicción completada con éxito (6 filas)`, código 0. Las predicciones son iguales a las de `v1`.
- **Persistencia:** después de los `--rm` no queda ningún contenedor, y en el equipo anfitrión siguen `models/v-docker/` (`modelo.joblib`, `metrics.json`, `metadata.json`) y `outputs/predicciones.csv`.
- **Entrada inválida** (CSV sin `pasajeros`): `ERROR: Al CSV de entrada le faltan las siguientes columnas requeridas: ['pasajeros']`, código 2, y no se crea el archivo de salida.
- **Versión repetida** (`--version v-docker` otra vez): `Error: La versión 'v-docker' ya existe en models/v-docker; use otro identificador con --version`, código 1.

## Pruebas

```bash
python -m pytest
```

Con Docker, montando los modelos, porque `tests/test_predict.py` lee `models/v1`:
```powershell
docker run --rm -v "${PWD}\models:/app/models:ro" modelos-fase2:1.0 python -m pytest -q
```

Resultado: **14 pruebas pasan**, tanto localmente como en el contenedor. No descargan datos.

**`tests/test_aplicacion.py`** usa datos crudos sintéticos de 300 filas, entrena en `tmp_path` y no modifica `models/`:

| Prueba | Qué verifica |
|---|---|
| `test_entrenamiento_y_almacenamiento` | `train.py` genera un `modelo.joblib` cargable, `metrics.json` (modelo y baseline) y `metadata.json` con esquema, semilla y tamaños |
| `test_carga_y_prediccion_conserva_resultados` | El modelo guardado predice igual que el modelo en memoria (tolerancia 1e-9), con una fila por fila y el orden conservado aunque las columnas vengan desordenadas |
| `test_valores_faltantes_se_imputan` | Las celdas nulas en `pasajeros`, `empresa` y `hora_salida_decimal` se procesan sin eliminar filas |
| `test_entrada_invalida_error_claro` (3 casos) | Una columna ausente, un texto en una columna numérica o un mes fuera de rango dan un mensaje claro, código ≠ 0 y no crean CSV de salida |
| `test_modelo_inexistente_error_claro` | Un modelo inexistente da un mensaje claro y código ≠ 0 |
| `test_versionamiento_no_sobrescribe` | Crear `v2` conserva `v1` byte a byte, ambas se cargan, y repetir `v1` se rechaza |
| `test_predictoras_sin_informacion_de_llegada` | Ninguna predictora deriva de la llegada |

**`tests/test_predict.py`** usa los modelos entregados y prueba la estructura de `models/v1/metadata.json`, el rechazo de columnas faltantes, el descarte de columnas extra, la predicción con `v1` en `tmp_path` y la ausencia de fuga de información.

## Resultados

Métricas sobre las 523.376 filas de test (en horas). La métrica principal es la MAE.

| Versión | Modelo | MAE | RMSE | R² | Baseline MAE | Reducción de la MAE |
|---|---|---|---|---|---|---|
| v1 | Random Forest (sin pasajeros) | **0.4322** (≈ 25,9 min) | 3.4886 | 0.5118 | 2.2104 | 80,4 % |
| v2 | Random Forest (sin pasajeros) | 0.4453 | 3.4932 | 0.5105 | 2.2104 | 79,9 % |

En la validación interna (MAE), el Random Forest sin pasajeros (0.4476) superó al Random Forest con pasajeros (0.4568), a la regresión lineal (0.5049) y al baseline (2.2142). Incluir `pasajeros` no mejoró la MAE, así que no se usa.

Las métricas de la Fase 1 (MAE 0.3622 h, R² 0.9520) **no son comparables**, porque allí se reemplazaban las duraciones extremas por la mediana de la ruta. Al conservar los retrasos reales, el RMSE aumenta y el R² baja: los errores grandes de unos pocos viajes muy largos dominan esas dos métricas, mientras que la MAE se mantiene baja.

## Limitaciones conocidas

- **Nulos fuera del rango pedido:** el dataset no cumple el intervalo de faltantes de 0,1–2 % que pide el proyecto. `fecha_hora_salida_origen` tiene 33,434 % de nulos (se completa con `fecha_salida`) y los demás campos con nulos están por debajo de 0,1 %.
- **Categorías desconocidas:** en una ruta conocida, una empresa nula o no vista puede producir predicciones atípicas. En el ejemplo, la última fila da unas 15,9 h para GUARNE.
- **Pruebas en Docker:** `tests/test_predict.py` depende de `models/v1`, por lo que en Docker hay que montar `models`.