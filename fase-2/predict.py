"""Genera predicciones con una versión guardada del modelo, sin reentrenar ni reajustar.

Ejemplos:
    python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
    python predict.py --model-version v2 --input data/ejemplo_prediccion.csv --output outputs/predicciones_v2.csv

Reglas de la entrada:
- Deben existir todas las columnas de `variables_entrada` del metadata.
- Una celda nula es válida (el pipeline la imputa); una columna ausente es un error.
- Las columnas adicionales se descartan. Las categorías no vistas se aceptan (el pipeline las
  codifica como desconocidas).
- Valores numéricos no convertibles, no enteros o fuera de rango producen un error.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
import joblib
import pandas as pd

# Evita errores de codificación al imprimir tildes.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

PREDICTION_COLUMN = "prediccion_duracion_horas"

# Rangos válidos de las variables numéricas: (mínimo, máximo, máximo incluido).
VALID_RANGES = {
    "hora_salida_decimal": (0, 24, False),
    "dia_semana": (0, 6, True),
    "mes": (1, 12, True),
    "pasajeros": (0, None, True),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Script de inferencia/predicción de la Fase 2."
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


def load_artifacts(base_dir: Path, version: str, model_path=None):
    """Carga la metadata y el modelo binario según la versión o la ruta especificada."""
    if model_path:
        model_path = Path(model_path)
        version_dir = model_path if model_path.is_dir() else model_path.parent
        if model_path.is_dir():
            model_path = version_dir / "modelo.joblib"
    else:
        version_dir = base_dir / version
        model_path = version_dir / "modelo.joblib"
    metadata_path = version_dir / "metadata.json"

    if not metadata_path.exists():
        print(
            f"ERROR: No se encontró el archivo de metadata en: {metadata_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not model_path.exists():
        print(
            f"ERROR: No se encontró el modelo guardado en: {model_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = joblib.load(model_path)
    return metadata, model


def _rows(mask, limit=5):
    """Números de fila (1 = primera fila de datos) para los mensajes de error."""
    rows = [str(int(i) + 1) for i in mask[mask].index[:limit]]
    return ", ".join(rows) + (" ..." if mask.sum() > limit else "")


def validate_types(df_clean: pd.DataFrame, feature_types: dict):
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
        df_clean[name] = values.astype(float)

    if errors:
        print("ERROR: Tipos de datos incompatibles en la entrada:\n  - " + "\n  - ".join(errors),
              file=sys.stderr)
        sys.exit(8)
    return df_clean


def validate_and_preprocess_input(df: pd.DataFrame, expected_features: list, feature_types: dict = None):
    """Valida la presencia de columnas y sus tipos, elimina columnas sobrantes y reordena segun el entrenamiento."""
    # 1. Comprobar columnas faltantes indispensables
    missing_cols = set(expected_features) - set(df.columns)
    if missing_cols:
        print(
            f"ERROR: Al CSV de entrada le faltan las siguientes columnas requeridas: {list(missing_cols)}",
            file=sys.stderr,
        )
        sys.exit(2)

    # 2. Filtrar columnas extra no esperadas y reordenar exactamente según el entrenamiento
    df_clean = df[expected_features].copy()

    # 3. Comprobar que no contenga la variable objetivo (evitar fugas o mal uso)
    target_candidates = ["duracion_viaje_horas", "target", "duracion"]
    for target in target_candidates:
        if target in df.columns and target not in expected_features:
            print(
                f"ADVERTENCIA: La columna objetivo '{target}' estaba en el CSV de entrada y fue descartada para inferencia."
            )

    # 4. Comprobar tipos y rangos según el esquema guardado en la versión
    if feature_types:
        df_clean = validate_types(df_clean, feature_types)

    return df_clean


def write_csv_atomic(df: pd.DataFrame, output_path: Path):
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


def main(argv=None):
    args = parse_args(argv)
    base_dir = Path(args.models_dir) if args.models_dir else Path(__file__).resolve().parent / "models"

    print(f"--- Iniciando Inferencia (Versión: {args.model or args.model_version}) ---")

    # 1. Cargar artefactos del modelo
    metadata, model = load_artifacts(base_dir, args.model_version, args.model)

    # Obtener el esquema de predictoras registrado durante el entrenamiento (train.py
    # lo guarda en "variables_entrada" como lista de {"nombre", "tipo"})
    feature_types = {
        item["nombre"]: item["tipo"] for item in metadata.get("variables_entrada", [])
    }
    expected_features = list(feature_types) or metadata.get(
        "features", metadata.get("predictors", None)
    )

    if not expected_features:
        # Fallback si en metadata no se listan explícitamente pero el pipeline las conserva
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
        else:
            print(
                "ERROR: No se pudo determinar la lista de columnas predictoras esperadas desde la metadata.",
                file=sys.stderr,
            )
            sys.exit(3)

    # 2. Leer archivo CSV de entrada
    input_path = Path(args.input)
    if not input_path.exists():
        print(
            f"ERROR: El archivo de entrada no existe: {input_path}",
            file=sys.stderr,
        )
        sys.exit(4)

    try:
        df_input = pd.read_csv(input_path)
    except Exception as e:
        print(
            f"ERROR: Falló la lectura del archivo CSV de entrada: {e}",
            file=sys.stderr,
        )
        sys.exit(5)

    if df_input.empty:
        print(
            "ERROR: El archivo CSV de entrada está vacío.", file=sys.stderr
        )
        sys.exit(6)

    # 3. Validar y alinear esquema de columnas
    df_prepared = validate_and_preprocess_input(df_input, expected_features, feature_types)

    # 4. Generar predicciones
    try:
        predictions = model.predict(df_prepared)
    except Exception as e:
        print(
            f"ERROR: Ocurrió un fallo durante la ejecución de las predicciones: {e}",
            file=sys.stderr,
        )
        sys.exit(7)

    # 5. Formatear y guardar resultados
    # Una fila de salida por fila de entrada, en el mismo orden
    df_output = df_prepared.copy()
    df_output[PREDICTION_COLUMN] = predictions

    output_path = Path(args.output)
    write_csv_atomic(df_output, output_path)

    print(
        f"Predicción completada con éxito ({len(df_output)} filas). Resultados guardados en: {output_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
