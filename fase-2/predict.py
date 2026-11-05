import argparse
import json
import sys
from pathlib import Path
import joblib
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script de inferencia/predicción de la Fase 2."
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
    return parser.parse_args()


def load_artifacts(base_dir: Path, version: str):
    """Carga la metadata y el modelo binario según la versión especificada."""
    version_dir = base_dir / "models" / version
    metadata_path = version_dir / "metadata.json"
    model_path = version_dir / "modelo.joblib"

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


def validate_and_preprocess_input(df: pd.DataFrame, expected_features: list):
    """Valida la presencia de columnas, elimina columnas sobrantes y reordena segun el entrenamiento."""
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

    return df_clean


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    print(f"--- Iniciando Inferencia (Versión: {args.model_version}) ---")

    # 1. Cargar artefactos del modelo
    metadata, model = load_artifacts(base_dir, args.model_version)

    # Obtener el esquema de predictoras registrado durante el entrenamiento
    expected_features = metadata.get(
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
    df_prepared = validate_and_preprocess_input(df_input, expected_features)

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
    df_output = df_prepared.copy()
    df_output["prediccion_duracion_horas"] = predictions

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, index=False)

    print(
        f"✅ Predicción completada con éxito. Resultados guardados en: {output_path}"
    )


if __name__ == "__main__":
    main()