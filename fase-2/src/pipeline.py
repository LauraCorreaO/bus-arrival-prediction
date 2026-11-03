"""Construcción de los modelos candidatos (port de build_model del notebook 05).

Cada modelo es un Pipeline completo: recibe las predictoras crudas, en cualquier orden,
y aprende la imputación y la codificación solo con los datos con los que se ajusta.
"""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder

from src.config import CATEGORICAL, DEFAULT_RF_PARAMS, DEFAULT_SEED
from src.features import prepare_features

ALGORITHMS = ("baseline", "linear", "random_forest")
ALGORITHM_NAMES = {
    "baseline": "Baseline (mediana)",
    "linear": "Regresión lineal",
    "random_forest": "Random Forest",
}


def build_model(algorithm, use_passengers=False, seed=DEFAULT_SEED, rf_params=None):
    """Devuelve un Pipeline sin entrenar para el algoritmo indicado.

    - baseline: DummyRegressor con la mediana del entrenamiento.
    - linear: one-hot de categorías, hora_entera, dia_semana y mes.
    - random_forest: codificación ordinal; las categorías no vistas se codifican como -1.
    """
    if algorithm == "baseline":
        return Pipeline([("regressor", DummyRegressor(strategy="median"))])

    if algorithm == "linear":
        transformers = [
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL + ["hora_entera", "dia_semana", "mes"]),
        ]
        if use_passengers:
            transformers.append(("passengers", SimpleImputer(strategy="median"), ["pasajeros"]))
        estimator = LinearRegression()
    elif algorithm == "random_forest":
        numeric = ["hora_salida_decimal", "dia_semana", "mes"]
        if use_passengers:
            numeric.append("pasajeros")
        transformers = [
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), CATEGORICAL),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ]
        estimator = RandomForestRegressor(
            **{**DEFAULT_RF_PARAMS, **(rf_params or {})}, n_jobs=-1, random_state=seed,
        )
    else:
        raise ValueError(f"Algoritmo desconocido: {algorithm}. Opciones: {ALGORITHMS}")

    # remainder="drop": las columnas que el modelo no usa se descartan
    return Pipeline([
        ("features", FunctionTransformer(prepare_features)),
        ("preprocessor", ColumnTransformer(transformers, remainder="drop")),
        ("regressor", estimator),
    ])
