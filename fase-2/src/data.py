"""Lectura, validación y preparación de los datos crudos (port del notebook 02 de la Fase 1).

Reglas que se conservan de la Fase 1 corregida:
- Hora, día de la semana y mes se derivan solo de la salida. La llegada se usa únicamente
  para calcular el objetivo, nunca para construir predictoras.
- Se excluyen filas sin salida, con salida fuera del período publicado, sin duración
  calculable o con duración no positiva. Las duraciones positivas largas se conservan.

La normalización de las categorías no se hace aquí sino dentro del pipeline del modelo
(src/features.py), para que entrenamiento y predicción apliquen exactamente la misma regla.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    PREDICTORS,
    RAW_REQUIRED_COLUMNS,
    TARGET,
    VALID_YEARS,
)


class DataValidationError(ValueError):
    """Error previsible en los datos de entrada (archivo o columnas)."""


def validate_columns(df, required_columns):
    """Lanza DataValidationError si falta alguna de las columnas requeridas."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise DataValidationError(f"Faltan columnas requeridas en los datos: {missing}")


def load_data(path):
    """Lee el CSV crudo (también .csv.gz) como texto y valida sus columnas."""
    path = Path(path)
    if not path.is_file():
        raise DataValidationError(f"No existe el archivo de datos: {path}")
    try:
        df = pd.read_csv(path, dtype=str)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise DataValidationError(f"No se pudo leer el archivo de datos {path}: {error}") from error

    validate_columns(df, RAW_REQUIRED_COLUMNS)
    return df


def departure_features(departure):
    """Deriva las predictoras temporales a partir de la fecha/hora de salida únicamente."""
    return pd.DataFrame({
        "hora_salida_decimal": departure.dt.hour + departure.dt.minute / 60 + departure.dt.second / 3600,
        "dia_semana": departure.dt.dayofweek,
        "mes": departure.dt.month,
    }, index=departure.index)


def prepare_data(df):
    """Construye predictoras y objetivo, y excluye filas inválidas.

    Devuelve el DataFrame con PREDICTORS + TARGET y un diccionario con el número de
    filas que cumple cada motivo de exclusión (un registro puede cumplir varios).
    """
    validate_columns(df, RAW_REQUIRED_COLUMNS)

    departure = pd.to_datetime(df["fecha_hora_salida_origen"].fillna(df["fecha_salida"]), errors="coerce")
    arrival = pd.to_datetime(df["hora_de_llegada"], errors="coerce")

    data = df[["nombre_sucursal", "empresa", "ruta_origen", "subregi_n", "clase_veh_culo"]].copy()
    data["pasajeros"] = pd.to_numeric(df["pasajeros"], errors="coerce")
    data = data.join(departure_features(departure))
    data[TARGET] = (arrival - departure).dt.total_seconds() / 3600

    no_departure = departure.isna()
    outside_period = ~departure.dt.year.between(*VALID_YEARS).fillna(False)
    no_duration = ~np.isfinite(data[TARGET])
    non_positive = data[TARGET].notna() & (data[TARGET] <= 0)
    exclude = no_departure | outside_period | no_duration | non_positive

    exclusions = {
        "filas_crudas": len(df),
        "sin_salida": int(no_departure.sum()),
        "salida_fuera_de_periodo": int(outside_period.sum()),
        "sin_duracion": int(no_duration.sum()),
        "duracion_no_positiva": int(non_positive.sum()),
        "total_excluidas": int(exclude.sum()),
    }

    data = data.loc[~exclude, PREDICTORS + [TARGET]].copy()
    data[["dia_semana", "mes"]] = data[["dia_semana", "mes"]].astype(int)
    exclusions["filas_conservadas"] = len(data)
    return data, exclusions


def split_data(data, test_size, seed):
    """Separa entrenamiento y prueba de forma aleatoria (cada viaje es independiente)."""
    return train_test_split(data, test_size=test_size, random_state=seed)
