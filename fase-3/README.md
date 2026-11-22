# Fase 3 · API REST

API FastAPI que estima la duración (horas) de un viaje hacia la Terminal de Transporte de Medellín.
Reutiliza el pipeline guardado en la fase 2 (`src/`, `models/<versión>/`).

| Ruta | Qué hace |
|---|---|
| `GET /health` | 200 y `{"estado": "en ejecución"}` |
| `POST /predict` | Recibe las 9 predictoras y devuelve `{"prediccion": <horas>, "version": "<id>"}` |
| `POST /train` | Entrena una versión nueva con los datos y la configuración de la fase 2 y devuelve su identificador |

Errores de `/predict`: 400 (cuerpo no JSON), 422 (falta un campo, tipo incompatible o fuera de rango; el mensaje nombra el campo),
503 (la versión configurada no existe), 500 (fallo del modelo). Un `null` es válido: el pipeline lo imputa (regla de la fase 2).

## Construir y ejecutar con Docker (desde `fase-3/`)

```powershell
docker build -t bus-api:fase3 .
```

```powershell
docker run --rm -p 8000:8000 -v "${PWD}/models:/app/models" -v "${PWD}/../fase-1/data/raw:/app/data" -v "${PWD}/registro:/app/registro" -e MODELO_VERSION=v1 bus-api:fase3
```

- Puerto: `8000` (el contenedor ejecuta `uvicorn api:app --host 0.0.0.0 --port 8000`).
- `models/` montado: versiones guardadas; `/train` escribe ahí las nuevas (persisten en el anfitrión).
- `/app/data` montado desde `fase-1/data/raw/`: `datos_completos.csv.gz` para `/train` (no se copia a la imagen).
- `registro/` montado: registro CSV de `/predict` (fase 4).
- La imagen no entrena durante el build.

Sin Docker: `python -m pip install -r requirements.txt` y luego `uvicorn api:app --host 0.0.0.0 --port 8000`.

## Seleccionar la versión del modelo

La API predice con la versión de la variable de entorno `MODELO_VERSION` (por defecto `v1`).
Las versiones disponibles son las carpetas de `models/` que contienen `modelo.joblib`:

```powershell
Get-ChildItem models
```

Para cambiar de versión, reinicien la API con otro valor (`-e MODELO_VERSION=v2` en Docker, o `$env:MODELO_VERSION="v2"` antes de `uvicorn`).
Si la versión no existe, `/predict` responde 503 con la lista de versiones disponibles.
Otras variables: `MODELS_DIR` (carpeta de versiones) y `REGISTRO_PATH` (archivo del registro).

## Llamar a la API

```powershell
curl.exe http://localhost:8000/health
```

```powershell
python cliente.py --sin-campo empresa
```

`cliente.py` consulta `/health`, envía dos viajes reales de `data/ejemplo_prediccion.csv` y uno sin `empresa`.
Ejemplo real (contenedor con `v1`, 22-11-2026):

```text
POST /predict
{"nombre_sucursal": "TERMINAL DEL SUR", "empresa": "TRANSPORTES JERICO", "ruta_origen": "JERICO", "subregi_n": "SUR",
 "clase_veh_culo": "AUTOMOVIL", "pasajeros": 1.0, "hora_salida_decimal": 14.0, "dia_semana": 6, "mes": 7}
-> 200 {"prediccion": 2.4945717155045317, "version": "v1"}

POST /predict (sin "empresa")
-> 422 {"error": "Entrada inválida", "mensaje": "Falta el campo obligatorio 'empresa'",
        "detalles": [{"campo": "empresa", "problema": "Falta el campo obligatorio 'empresa'"}]}
```

### `/train`

El cuerpo es opcional. Sin cuerpo (o con `{}`) entrena con `data/datos_completos.csv.gz`, la variable objetivo
`duracion_viaje_horas`, la semilla 42 y los hiperparámetros de `src/config.py`, y genera el siguiente identificador libre `vN`:

```powershell
curl.exe -X POST http://localhost:8000/train
```

```text
201 {"version": "v3", "mensaje": "Versión 'v3' entrenada y guardada correctamente"}
```

Ejecución real en Docker (22-11-2026): 11 min con las 2 616 877 filas. Se creó `models/v3/` con los tres artefactos;
v1 y v2 no cambiaron. Con los mismos datos y semilla, v3 reproduce las métricas de v1 (Random Forest sin pasajeros,
MAE 0,4322 h, RMSE 3,4886 h, R² 0,5118 en test), lo que confirma que el entrenamiento es reproducible.

Campos opcionales: `version` (identificador propio), `data`, `seed`, `test_size`, `val_size`, `n_estimators`,
`min_samples_leaf`, `max_samples` y `max_rows` (muestra para un entrenamiento rápido, p. ej. `{"version": "v3", "max_rows": 5000}`).
Con los datos completos (2,6 M filas) tarda varios minutos; la respuesta llega cuando los artefactos están guardados.
Un identificador existente se rechaza con 409 y nunca se sobrescribe; si falta el archivo de datos responde 400.
La predicción sigue usando `MODELO_VERSION` hasta que se cambie.

**Datos para `/train`.** `datos_completos.csv.gz` no se versiona en `fase-3/`: está en `fase-1/data/raw/`. Cópienlo
(`Copy-Item ../fase-1/data/raw/datos_completos.csv.gz data/`) o, en Docker, monten esa carpeta con
aw\datos_completos.csv.gz data\`) o, en Docker, monten esa carpeta con
`-v "${PWD}/../fase-1/data/raw:/app/data"`.

## Pruebas (desde `fase-3/`)

```powershell
python -m pytest
```

Tres casos en `tests/test_api.py`: `/health` responde 200 y el estado; una entrada válida devuelve predicción y versión;
una entrada sin `empresa` se rechaza con 422 y el nombre del campo. Usan un modelo diminuto creado en una carpeta temporal
(no cargan `v1` ni entrenan con los datos completos).

## Gestión de versiones del modelo

Cada versión (creada con `train.py` o con `POST /train`) se guarda en `fase-3/models/<versión>/` con tres artefactos:
`modelo.joblib` (pipeline completo), `metrics.json` (MAE, RMSE y R² en test) y `metadata.json` (datos, semilla, parámetros y
variables de entrada). Si el identificador ya existe la operación se rechaza y no se sobrescribe nada.

| Versión | Origen | Uso |
|---|---|---|
| `v1` | Fase 2, datos completos | Activa por defecto (`MODELO_VERSION=v1`) |
| `v2` | Fase 2, datos completos | Alternativa |
| `v3` | `POST /train` sin cuerpo (datos completos, semilla 42), 22-11-2026 | Reentrenamiento de verificación, no activa |

Ver las versiones y sus métricas (desde `fase-3/`):

```powershell
Get-ChildItem models
```

```powershell
Get-Content models/v1/metrics.json
```

Para seleccionar una, vean «Seleccionar la versión del modelo»: se cambia `MODELO_VERSION` y se reinicia la API.
Para volver a la anterior basta con reiniciar con el valor previo (por ejemplo `-e MODELO_VERSION=v1`).

Entrenar por línea de comandos:

```powershell
python train.py --data data/datos_completos.csv.gz --version v5 --seed 42
```
