"""API REST del modelo de duración de viaje.

Inicio (desde fase-3):
    uvicorn api:app --host 0.0.0.0 --port 8000

Rutas:
    GET  /health   indica que la API está en ejecución.
    POST /predict  recibe un viaje (las 9 predictoras) y devuelve la duración estimada en horas.
    POST /train    entrena una versión nueva (cuerpo opcional; sin "version" genera el siguiente vN) sin sobrescribir.

Errores de /predict y /train (cuerpo {"error", "mensaje", "detalles": [{"campo", "problema"}]}):
    400  el cuerpo no es un objeto JSON válido o no se encuentra el archivo de datos.
    409  el identificador de versión ya existe (no se sobrescribe).
    422  falta un campo obligatorio, tiene un tipo incompatible o está fuera de rango.
    500  el modelo falló al predecir.
    503  la versión configurada (MODELO_VERSION) no existe o no se pudo cargar.

Configuración por variables de entorno:
    MODELO_VERSION  versión con la que predice /predict (por defecto v1).
    MODELS_DIR      carpeta de versiones (por defecto models/ junto a api.py).
    REGISTRO_PATH   CSV del registro de /predict (por defecto registro/solicitudes.csv).

/predict reutiliza src/prediction.py, el mismo código de predict.py: carga la versión guardada
y aplica su pipeline (normalización, imputación y codificación) sin reentrenar.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

from src.config import DEFAULT_RF_PARAMS, DEFAULT_SEED, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE, PREDICTORS
from src.data import DataValidationError
from src.prediction import (InputValidationError, MissingColumnsError, ModelNotAvailableError, PredictionError,
                            load_artifacts, predict_dataframe)
from src.registro import registrar
from src.versioning import VERSION_PATTERN
from src.versioning import VersionExistsError
from train import train_model

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_DATA_PATH = BASE_DIR / "data" / "datos_completos.csv.gz"
DEFAULT_VERSION = "v1"

# Una categoría se envía como texto y un número como número. Todas las claves son obligatorias,
# pero aceptan null: el pipeline guardado imputa el valor faltante (regla de la fase 2).
Texto = Optional[StrictStr]
Decimal = Optional[StrictFloat | StrictInt]
Entero = Optional[StrictInt]


class Viaje(BaseModel):
    """Predictoras de un viaje, conocidas al momento de la salida."""

    model_config = ConfigDict(
        extra="ignore",   # las claves adicionales se descartan, igual que las columnas extra en predict.py
        json_schema_extra={"examples": [{
            "nombre_sucursal": "TERMINAL DEL SUR",
            "empresa": "TRANSPORTES JERICO",
            "ruta_origen": "JERICO",
            "subregi_n": "SUR",
            "clase_veh_culo": "AUTOMOVIL",
            "pasajeros": 1,
            "hora_salida_decimal": 14.0,
            "dia_semana": 6,
            "mes": 7,
        }]},
    )

    nombre_sucursal: Annotated[Texto, Field(description="Terminal de llegada, p. ej. TERMINAL DEL SUR")]
    empresa: Annotated[Texto, Field(description="Empresa transportadora")]
    ruta_origen: Annotated[Texto, Field(description="Municipio de origen del viaje")]
    subregi_n: Annotated[Texto, Field(description="Subregión de origen")]
    clase_veh_culo: Annotated[Texto, Field(description="Clase de vehículo, p. ej. BUS")]
    pasajeros: Annotated[Decimal, Field(ge=0, description="Número de pasajeros")]
    hora_salida_decimal: Annotated[Decimal, Field(ge=0, lt=24, description="Hora de salida en [0, 24), p. ej. 14.5 = 14:30")]
    dia_semana: Annotated[Entero, Field(ge=0, le=6, description="Día de la semana de la salida: 0 = lunes ... 6 = domingo")]
    mes: Annotated[Entero, Field(ge=1, le=12, description="Mes de la salida: 1 ... 12")]


# El esquema de la API debe coincidir con las predictoras usadas al entrenar
assert list(Viaje.model_fields) == PREDICTORS, "Viaje no coincide con src.config.PREDICTORS"


class Prediccion(BaseModel):
    prediccion: float = Field(description="Duración estimada del viaje, en horas")
    version: str = Field(description="Versión del modelo que produjo la predicción")


class EntrenamientoSolicitud(BaseModel):
    """Parámetros para entrenar una nueva versión del modelo."""

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={"examples": [{
            "version": "v4",
            "data": "data/datos_completos.csv.gz",
            "seed": 42,
            "max_rows": 5000,
        }]},
    )

    version: Annotated[Optional[StrictStr], Field(default=None, min_length=1,
                       description="Identificador de la nueva versión; si se omite se genera el siguiente vN libre")] = None
    data: Annotated[Optional[StrictStr], Field(default=None, description="Ruta al CSV crudo (por defecto data/datos_completos.csv.gz)")] = None
    seed: Annotated[StrictInt, Field(default=DEFAULT_SEED, description="Semilla aleatoria")] = DEFAULT_SEED
    test_size: Annotated[StrictFloat, Field(gt=0, lt=1, default=DEFAULT_TEST_SIZE)] = DEFAULT_TEST_SIZE
    val_size: Annotated[StrictFloat, Field(gt=0, lt=1, default=DEFAULT_VAL_SIZE)] = DEFAULT_VAL_SIZE
    n_estimators: Annotated[StrictInt, Field(ge=1, default=DEFAULT_RF_PARAMS["n_estimators"])] = DEFAULT_RF_PARAMS["n_estimators"]
    min_samples_leaf: Annotated[StrictInt, Field(ge=1, default=DEFAULT_RF_PARAMS["min_samples_leaf"])] = DEFAULT_RF_PARAMS["min_samples_leaf"]
    max_samples: Annotated[StrictFloat, Field(gt=0, le=1, default=DEFAULT_RF_PARAMS["max_samples"])] = DEFAULT_RF_PARAMS["max_samples"]
    max_rows: Annotated[Optional[StrictInt], Field(ge=100, default=None, description="Muestra opcional para entrenamientos rápidos")] = None


class EntrenamientoRespuesta(BaseModel):
    version: str = Field(description="Identificador de la versión entrenada y guardada")
    mensaje: str = Field(description="Confirmación del entrenamiento")


app = FastAPI(
    title="Duración de viajes - Terminal de Transporte",
    description="Estima la duración (horas) de un viaje hacia la Terminal de Transporte de Medellín.",
    version="3.0",
)


def active_version():
    """Versión con la que predice la API: variable de entorno MODELO_VERSION (por defecto v1)."""
    return os.environ.get("MODELO_VERSION", DEFAULT_VERSION)


def models_dir():
    """Carpeta de versiones: variable de entorno MODELS_DIR (por defecto models/ junto a api.py)."""
    return Path(os.environ.get("MODELS_DIR", MODELS_DIR))


def get_model(version):
    return _load_model(str(models_dir()), version)


@lru_cache(maxsize=4)
def _load_model(directory, version):
    """Carga una versión una sola vez y la conserva en memoria. Un fallo no queda en caché."""
    if not VERSION_PATTERN.match(version):
        raise ModelNotAvailableError(f"Identificador de versión inválido en MODELO_VERSION: {version!r}")
    return load_artifacts(Path(directory) / version)


@app.middleware("http")
async def registrar_predicciones(request: Request, call_next):
    """Fase 4: una fila del registro por cada intento de /predict, incluidos los rechazos.

    El middleware envuelve también a los manejadores de error (RequestValidationError incluido),
    así que el 400/422 que devuelven queda registrado con el código real de la respuesta.
    """
    if request.url.path != "/predict":
        return await call_next(request)
    try:
        response = await call_next(request)
    except Exception:
        registrar(500, getattr(request.state, "version", None))
        raise
    registrar(response.status_code, getattr(request.state, "version", None),
              getattr(request.state, "prediccion", None) if response.status_code == 200 else None)
    return response


def error_response(status_code, error, detalles):
    """Respuesta de error común: un mensaje legible y un detalle por campo."""
    mensaje = "; ".join(detalle["problema"] for detalle in detalles)
    return JSONResponse(status_code=status_code,
                        content={"error": error, "mensaje": mensaje, "detalles": detalles})


# Errores del cuerpo completo (no de un campo): se responden con 400
BODY_ERRORS = {
    "json_invalid": "El cuerpo de la solicitud no es un JSON válido",
    "model_attributes_type": "El cuerpo debe ser un objeto JSON con las predictoras del viaje",
    "missing": "Falta el cuerpo de la solicitud: envíe un objeto JSON con las predictoras del viaje",
}


def describe_field_error(field, error):
    """Traduce un error de Pydantic a un mensaje en español que nombra el campo."""
    kind, ctx = error["type"], error.get("ctx", {})
    if kind == "missing":
        return f"Falta el campo obligatorio '{field}'"
    if kind == "string_type":
        return f"'{field}' debe ser texto o null"
    if kind == "int_type":
        return f"'{field}' debe ser un número entero o null"
    if kind == "float_type":
        return f"'{field}' debe ser un número o null"
    if kind == "greater_than_equal":
        return f"'{field}' debe ser mayor o igual a {ctx['ge']}"
    if kind == "less_than":
        return f"'{field}' debe ser menor que {ctx['lt']}"
    if kind == "less_than_equal":
        return f"'{field}' debe ser menor o igual a {ctx['le']}"
    return f"'{field}' no es válido: {error['msg']}"


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    body_error = next((e for e in errors if len(e["loc"]) == 1 or e["type"] == "json_invalid"), None)
    if body_error is not None:
        problem = BODY_ERRORS.get(body_error["type"], f"Cuerpo inválido: {body_error['msg']}")
        return error_response(400, "Solicitud inválida", [{"campo": None, "problema": problem}])

    # Un campo con tipo número acepta float o int; Pydantic informa un error por cada opción.
    # Se deja un solo mensaje por campo.
    detalles = {}
    for error in errors:
        field = str(error["loc"][1])
        if field not in detalles:
            kind = "float_type" if tuple(error["loc"][2:]) in (("float",), ("int",)) else error["type"]
            detalles[field] = describe_field_error(field, {**error, "type": kind})
    return error_response(422, "Entrada inválida",
                          [{"campo": campo, "problema": problema} for campo, problema in detalles.items()])


@app.exception_handler(InputValidationError)
async def input_error_handler(request: Request, exc: InputValidationError):
    campos = exc.columns if isinstance(exc, MissingColumnsError) else [None] * len(exc.errors)
    return error_response(422, "Entrada inválida",
                          [{"campo": campo, "problema": problema} for campo, problema in zip(campos, exc.errors)])


@app.exception_handler(ModelNotAvailableError)
async def model_unavailable_handler(request: Request, exc: ModelNotAvailableError):
    disponibles = sorted(p.name for p in models_dir().glob("*") if (p / "modelo.joblib").is_file())
    return error_response(503, "Modelo no disponible", [{"campo": None, "problema": (
        f"La versión configurada '{active_version()}' no está disponible ({exc}). "
        f"Versiones disponibles: {', '.join(disponibles) or 'ninguna'}. "
        "Seleccione una con la variable de entorno MODELO_VERSION y reinicie la API.")}])


@app.exception_handler(PredictionError)
async def prediction_error_handler(request: Request, exc: PredictionError):
    return error_response(500, "No se pudo generar la predicción", [{"campo": None, "problema": str(exc)}])


@app.exception_handler(VersionExistsError)
async def version_exists_handler(request: Request, exc: VersionExistsError):
    return error_response(409, "Versión existente", [{"campo": "version", "problema": str(exc)}])


@app.get("/health")
def health():
    return {"estado": "en ejecución"}


@app.post("/predict", response_model=Prediccion)
def predict(viaje: Viaje, request: Request):
    version = active_version()
    model, metadata = get_model(version)
    request.state.version = metadata.get("version", version)
    _, predictions = predict_dataframe(model, metadata, pd.DataFrame([viaje.model_dump()]))
    request.state.prediccion = float(predictions[0])
    return Prediccion(prediccion=request.state.prediccion, version=request.state.version)


def next_version_id(directory):
    """Siguiente identificador vN que no existe en la carpeta de versiones (v1, v2 -> v3)."""
    directory = Path(directory)
    numeros = [int(p.name[1:]) for p in directory.glob("v*") if p.name[1:].isdigit()] if directory.is_dir() else []
    return f"v{max(numeros, default=0) + 1}"


@app.post("/train", response_model=EntrenamientoRespuesta, status_code=201)
def train_endpoint(solicitud: Optional[EntrenamientoSolicitud] = None):
    # El cuerpo es opcional: sin cuerpo (o con {}) se usan los datos, la semilla y los
    # parámetros de src/config.py y se genera un identificador disponible.
    solicitud = solicitud or EntrenamientoSolicitud()
    data_path = Path(solicitud.data) if solicitud.data else DEFAULT_DATA_PATH
    if not data_path.is_absolute():
        data_path = BASE_DIR / data_path

    if not data_path.is_file():
        return error_response(
            400,
            "Datos no encontrados",
            [{"campo": "data", "problema": f"No existe el archivo de datos: {data_path}"}],
        )

    try:
        version_creada = train_model(
            data_path=data_path,
            version=solicitud.version or next_version_id(models_dir()),
            output_dir=models_dir(),
            seed=solicitud.seed,
            test_size=solicitud.test_size,
            val_size=solicitud.val_size,
            n_estimators=solicitud.n_estimators,
            min_samples_leaf=solicitud.min_samples_leaf,
            max_samples=solicitud.max_samples,
            max_rows=solicitud.max_rows,
        )
    except ValueError as exc:
        campo = "version" if "Identificador de versión inválido" in str(exc) else None
        return error_response(422, "Parámetros de entrenamiento inválidos", [{"campo": campo, "problema": str(exc)}])
    except DataValidationError as exc:
        return error_response(422, "Datos de entrenamiento inválidos", [{"campo": "data", "problema": str(exc)}])

    return EntrenamientoRespuesta(
        version=version_creada,
        mensaje=f"Versión '{version_creada}' entrenada y guardada correctamente",
    )