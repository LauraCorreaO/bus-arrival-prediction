"""Validación y preparación de entradas para predicción."""

import sys
from collections.abc import Sequence

import pandas as pd


def validate_and_preprocess_input(
    data: pd.DataFrame, expected_features: Sequence[str]
) -> pd.DataFrame:
    """Valida las columnas requeridas y descarta las columnas adicionales."""
    missing_features = [
        feature for feature in expected_features if feature not in data.columns
    ]
    if missing_features:
        print(
            "Error: faltan columnas requeridas: " + ", ".join(missing_features),
            file=sys.stderr,
        )
        raise SystemExit(2)

    return data.loc[:, list(expected_features)].copy()
