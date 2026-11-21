"""Carga de una versión guardada y predicción sobre datos nuevos, sin reentrenar ni reajustar.

Lo usan predict.py (línea de comandos) y api.py (/predict). Las funciones no terminan el
proceso: lanzan excepciones que cada interfaz traduce a su forma de error (código de salida
o respuesta HTTP).

Reglas de la entrada:
- Deben existir todas las columnas de `variables_entrada` del metadata.
- Una celda nula es válida (el pipeline la imputa); una columna ausente es un error.
- Las columnas adicionales se descartan. Las categorías no vistas se aceptan (el pipeline las
  codifica como desconocidas).
- Valores numéricos no convertibles, no enteros o fuera de rango producen un error.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from src.versioning import METADATA_FILE, MODEL_FILE

# Rangos válidos de las variables numéricas: (mínimo, máximo, máximo incluido).
VALID_RANGES = {
    "hora_salida_decimal": (0, 24, False),
    "dia_semana": (0, 6, True),
    "mes": (1, 12, True),
    "pasajeros": (0, None, True),
}


class PredictionError(Exception):
    """Error previsible al predecir. exit_code es el código que usa predict.py."""

    exit_code = 1


class ModelNotAvailableError(PredictionError):
    """No existe la versión, su modelo o su metadata."""

    exit_code = 1


class InputValidationError(PredictionError):
    """La entrada no cumple el esquema guardado. errors lista un mensaje por problema."""

    exit_code = 8

    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors or [message]


class MissingColumnsError(InputValidationError):
    """Faltan columnas obligatorias. columns lista sus nombres."""

    exit_code = 2

    def __init__(self, columns):
        self.columns = sorted(columns)
        super().__init__(f"Faltan las siguientes columnas requeridas: {self.columns}",
                         [f"Falta el campo obligatorio '{column}'" for column in self.columns])


class PredictionFailedError(PredictionError):
    """El modelo falló al predecir con una entrada que sí pasó la validación."""

    exit_code = 7


def resolve_version_dir(models_dir=None, version=None, model_path=None):
    """Carpeta de la versión a partir de una ruta explícita o de models_dir/version."""
    if model_path:
        model_path = Path(model_path)
        return model_path if model_path.is_dir() else model_path.parent
    return Path(models_dir) / version


def load_artifacts(version_dir):
    """Carga el modelo y la metadata de una versión. Devuelve (modelo, metadata)."""
    version_dir = Path(version_dir)
    metadata_path = version_dir / METADATA_FILE
    model_path = version_dir / MODEL_FILE

    if not metadata_path.is_file():
        raise ModelNotAvailableError(f"No se encontró el archivo de metadata en: {metadata_path}")
    if not model_path.is_file():
        raise ModelNotAvailableError(f"No se encontró el modelo guardado en: {model_path}")

    try:
        with open(metadata_path, encoding="utf-8") as file:
            metadata = json.load(file)
        model = joblib.load(model_path)
    except Exception as error:
        raise ModelNotAvailableError(f"No se pudo cargar la versión en {version_dir}: {error}") from error
    return model, metadata


def input_schema(model, metadata):
    """Nombres y tipos de las predictoras, en el orden usado al entrenar."""
    feature_types = {item["nombre"]: item["tipo"] for item in metadata.get("variables_entrada", [])}
    if feature_types:
        return list(feature_types), feature_types
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_), {}
    raise ModelNotAvailableError(
        "No se pudo determinar la lista de columnas predictoras esperadas desde la metadata."
    )


def _rows(mask, limit=5):
    """Números de fila (1 = primera fila de datos) para los mensajes de error."""
    rows = [str(int(i) + 1) for i in mask[mask].index[:limit]]
    return ", ".join(rows) + (" ..." if mask.sum() > limit else "")


def validate_types(df_clean, feature_types):
    """Comprueba tipos y rangos de las columnas numéricas. Los nulos se permiten (se imputan)."""
    errors = []
    for name, kind in feature_types.items():
        if kind == "string" or name not in df_clean.columns:
            continue
        values = pd.to_numeric(df_clean[name], errors="coerce")
        invalid = values.isna() & df_clean[name].notna()
        if invalid.any():
            errors.append(f"'{name}' debe ser numérica; valores inválidos en las filas {_rows(invalid)}")
            continue
        if kind == "int" and (values.notna() & (values % 1 != 0)).any():
            errors.append(f"'{name}' debe ser entera; hay valores no enteros en las filas "
                          f"{_rows(values.notna() & (values % 1 != 0))}")
            continue
        low, high, high_inclusive = VALID_RANGES.get(name, (None, None, True))
        out = pd.Series(False, index=values.index)
        if low is not None:
            out |= values < low
        if high is not None:
            out |= (values > high) if high_inclusive else (values >= high)
        if out.any():
            errors.append(f"'{name}' fuera de rango en las filas {_rows(out)}")
            continue
        df_clean[name] = values.astype(float) if kind == "float" else values

    if errors:
        raise InputValidationError("Tipos de datos incompatibles en la entrada: " + "; ".join(errors), errors)
    return df_clean


def validate_input(df, expected_features, feature_types=None):
    """Valida columnas y tipos, descarta columnas sobrantes y ordena según el entrenamiento."""
    missing = set(expected_features) - set(df.columns)
    if missing:
        raise MissingColumnsError(missing)

    df_clean = df[expected_features].copy()
    if feature_types:
        df_clean = validate_types(df_clean, feature_types)
    return df_clean


def predict_dataframe(model, metadata, df):
    """Valida la entrada y devuelve (entrada_preparada, predicciones), una por fila y en orden."""
    if df.empty:
        raise InputValidationError("La entrada no contiene filas para predecir.")

    expected_features, feature_types = input_schema(model, metadata)
    df_prepared = validate_input(df.reset_index(drop=True), expected_features, feature_types)
    try:
        predictions = model.predict(df_prepared)
    except Exception as error:
        raise PredictionFailedError(f"Ocurrió un fallo durante la ejecución de las predicciones: {error}") from error
    return df_prepared, predictions
