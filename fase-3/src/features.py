"""Transformaciones sin estado que forman parte del pipeline del modelo.

Al ir dentro del Pipeline se guardan con el modelo y se aplican igual en entrenamiento
y en predicción. Son funciones de módulo (no lambdas) para que joblib pueda serializarlas.
"""

from src.config import CATEGORICAL

UNKNOWN_CATEGORY = "DESCONOCIDA"
DESTINATION_SUFFIX = r"\s*-\s*MEDELLIN$"   # "RIONEGRO-MEDELLIN" -> "RIONEGRO"


def normalize_categoricals(X):
    """Hace explícitos los nulos y normaliza el texto de las variables categóricas."""
    X = X.copy()
    for column in CATEGORICAL:
        X[column] = X[column].fillna(UNKNOWN_CATEGORY).astype(str).str.upper().str.strip()
    X["ruta_origen"] = X["ruta_origen"].str.replace(DESTINATION_SUFFIX, "", regex=True).str.strip()
    return X


def add_hora_entera(X):
    """Agrega la hora de salida truncada, que la regresión lineal usa como categoría."""
    X = X.copy()
    X["hora_entera"] = X["hora_salida_decimal"].fillna(0).astype(int)
    return X


def prepare_features(X):
    """Paso inicial del pipeline: normalización de categorías y hora_entera."""
    return add_hora_entera(normalize_categoricals(X))
