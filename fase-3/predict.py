"""Genera predicciones con una versión guardada del modelo, sin reentrenar ni reajustar.

Ejemplos:
    python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
    python predict.py --model-version v2 --input data/ejemplo_prediccion.csv --output outputs/predicciones_v2.csv

La validación y la predicción están en src/prediction.py (las reutiliza la API). Este script
solo lee el CSV, escribe el resultado y convierte los errores en mensajes con código de salida
distinto de cero. Las reglas de la entrada se describen en src/prediction.py.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.config import TARGET
from src.prediction import (
    PredictionError,
    load_artifacts,
    predict_dataframe,
    resolve_version_dir,
)

# Evita errores de codificación al imprimir tildes.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

PREDICTION_COLUMN = "prediccion_duracion_horas"


class InputFileNotFoundError(PredictionError):
    exit_code = 4


class InputFileReadError(PredictionError):
    exit_code = 5


class EmptyInputError(PredictionError):
    exit_code = 6


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Genera predicciones con una versión guardada del modelo."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ruta al modelo.joblib o a la carpeta de la versión (ej. models/v1/modelo.joblib). "
        "Si se indica, tiene prioridad sobre --model-version.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Carpeta de versiones usada con --model-version. Por defecto: models/ junto a predict.py",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Ruta al archivo CSV con las características de entrada para predecir.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Ruta donde se guardará el archivo CSV con las predicciones.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1",
        help="Versión del modelo a utilizar (ej. v1, v2). Por defecto: v1",
    )
    return parser.parse_args(argv)


def read_input(input_path):
    """Lee el CSV de entrada; lanza un PredictionError si no existe, no se puede leer o está vacío."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise InputFileNotFoundError(f"El archivo de entrada no existe: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError as error:
        raise EmptyInputError("El archivo CSV de entrada está vacío.") from error
    except Exception as error:
        raise InputFileReadError(f"Falló la lectura del archivo CSV de entrada: {error}") from error
    if df.empty:
        raise EmptyInputError("El archivo CSV de entrada está vacío.")
    return df


def write_csv_atomic(df, output_path):
    """Escribe en un temporal y lo renombra: si algo falla no queda un CSV que aparente ser válido."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pred-", suffix=".csv", dir=output_path.parent)
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, output_path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def run(args):
    base_dir = Path(args.models_dir) if args.models_dir else Path(__file__).resolve().parent / "models"
    print(f"--- Iniciando Inferencia (Versión: {args.model or args.model_version}) ---")

    model, metadata = load_artifacts(resolve_version_dir(base_dir, args.model_version, args.model))
    df_input = read_input(args.input)
    if TARGET in df_input.columns:
        print(f"ADVERTENCIA: La columna objetivo '{TARGET}' estaba en el CSV de entrada y fue descartada para inferencia.")

    # Una fila de salida por fila de entrada, en el mismo orden
    df_output, predictions = predict_dataframe(model, metadata, df_input)
    df_output[PREDICTION_COLUMN] = predictions

    output_path = Path(args.output)
    write_csv_atomic(df_output, output_path)
    print(f"Predicción completada con éxito ({len(df_output)} filas). Resultados guardados en: {output_path}")


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except PredictionError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return error.exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
