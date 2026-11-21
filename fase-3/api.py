"""API REST del modelo de duración de viaje.

Inicio (desde fase-3):
    uvicorn api:app --host 0.0.0.0 --port 8000

Rutas:
    GET  /health   indica que la API está en ejecución.
    POST /predict  recibe un viaje (las 9 predictoras) y devuelve la duración estimada en horas.

/predict reutiliza src/prediction.py, el mismo código de predict.py: carga la versión guardada
y aplica su pipeline (normalización, imputación y codificación) sin reentrenar.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr

from src.config import PREDICTORS
from src.prediction import load_artifacts, predict_dataframe

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
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


app = FastAPI(
    title="Duración de viajes - Terminal de Transporte",
    description="Estima la duración (horas) de un viaje hacia la Terminal de Transporte de Medellín.",
    version="3.0",
)


def active_version():
    """Versión con la que predice la API."""
    return DEFAULT_VERSION


@lru_cache(maxsize=4)
def get_model(version):
    """Carga una versión una sola vez y la conserva en memoria. Un fallo no queda en caché."""
    return load_artifacts(MODELS_DIR / version)


@app.get("/health")
def health():
    return {"estado": "en ejecución"}


@app.post("/predict", response_model=Prediccion)
def predict(viaje: Viaje):
    version = active_version()
    model, metadata = get_model(version)
    _, predictions = predict_dataframe(model, metadata, pd.DataFrame([viaje.model_dump()]))
    return Prediccion(prediccion=float(predictions[0]), version=metadata.get("version", version))
