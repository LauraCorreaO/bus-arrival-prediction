"""Entrena una versión del modelo de duración de viaje y la guarda en <output-dir>/<version>/.

Ejemplo (desde fase-2):
    python train.py --data data/datos_completos.csv.gz --target duracion_viaje_horas --output-dir models --version v1 --seed 42
"""

import argparse
import hashlib
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.config import (
    DEFAULT_RF_PARAMS,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    FEATURE_TYPES,
    PASSENGERS_MIN_IMPROVEMENT,
    PREDICTORS,
    TARGET,
    VALID_YEARS,
)
from src.data import DataValidationError, load_data, prepare_data, split_data
from src.evaluation import compute_metrics, select_model
from src.pipeline import build_model
from src.versioning import VersionExistsError, ensure_version_available, git_commit, save_version

DATA_SOURCE = "https://www.datos.gov.co/Transporte/Llegada-y-Salidas-de-Veh-culos-de-la-Terminal-de-T/pfsr-mdyi"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de duración de viaje (regresión) y guarda una versión nueva.",
    )
    parser.add_argument("--data", required=True,
                        help="CSV crudo descargado de datos.gov.co (admite .csv.gz)")
    parser.add_argument("--target", default=TARGET,
                        help=f"Variable objetivo; se calcula como llegada - salida (por defecto: {TARGET})")
    parser.add_argument("--output-dir", default="models", help="Carpeta donde se crean las versiones (por defecto: models)")
    parser.add_argument("--version", required=True, help="Identificador único de la versión, por ejemplo v1")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Semilla aleatoria (por defecto: {DEFAULT_SEED})")
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE,
                        help=f"Proporción de prueba (por defecto: {DEFAULT_TEST_SIZE})")
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE,
                        help=f"Proporción de train usada como validación interna (por defecto: {DEFAULT_VAL_SIZE})")
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_RF_PARAMS["n_estimators"],
                        help=f"Árboles del Random Forest (por defecto: {DEFAULT_RF_PARAMS['n_estimators']})")
    parser.add_argument("--min-samples-leaf", type=int, default=DEFAULT_RF_PARAMS["min_samples_leaf"],
                        help=f"Mínimo de filas por hoja (por defecto: {DEFAULT_RF_PARAMS['min_samples_leaf']})")
    parser.add_argument("--max-samples", type=float, default=DEFAULT_RF_PARAMS["max_samples"],
                        help=f"Fracción de filas por árbol (por defecto: {DEFAULT_RF_PARAMS['max_samples']})")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Usa una muestra aleatoria de este tamaño (para pruebas rápidas; por defecto, todas las filas)")
    args = parser.parse_args(argv)

    if args.target != TARGET:
        parser.error(f"--target debe ser '{TARGET}': es la única variable objetivo que se deriva de los datos crudos")
    for name in ("test_size", "val_size", "max_samples"):
        if not 0 < getattr(args, name) < 1 and not (name == "max_samples" and getattr(args, name) == 1):
            parser.error(f"--{name.replace('_', '-')} debe estar entre 0 y 1")
    if args.max_rows is not None and args.max_rows < 100:
        parser.error("--max-rows debe ser al menos 100")
    return args


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    started = time.time()
    ensure_version_available(args.output_dir, args.version)
    rf_params = {"n_estimators": args.n_estimators, "min_samples_leaf": args.min_samples_leaf,
                 "max_samples": args.max_samples}

    print(f"Leyendo datos: {args.data}")
    data, exclusions = prepare_data(load_data(args.data))
    if args.max_rows is not None and args.max_rows < len(data):
        data = data.sample(args.max_rows, random_state=args.seed)
    print(f"Filas utilizables: {len(data):,} (excluidas: {exclusions['total_excluidas']:,})")

    # La partición se hace antes de ajustar cualquier transformación
    train, test = split_data(data, args.test_size, args.seed)
    print(f"Entrenamiento: {len(train):,} | prueba: {len(test):,}")

    print("Selección con validación interna (solo train):")
    selection = select_model(train, args.val_size, args.seed, rf_params)
    print(f"Modelo seleccionado: {selection['modelo']}")

    # Reajuste con todo train y una única evaluación en test, junto al baseline
    X_train, y_train = train[PREDICTORS], train[TARGET]
    X_test, y_test = test[PREDICTORS], test[TARGET]
    model = build_model(selection["algoritmo"], selection["usa_pasajeros"], seed=args.seed, rf_params=rf_params)
    model.fit(X_train, y_train)
    baseline = build_model("baseline").fit(X_train, y_train)
    model_metrics = compute_metrics(y_test, model.predict(X_test))
    baseline_metrics = compute_metrics(y_test, baseline.predict(X_test))

    metrics = {
        "version": args.version,
        "metrica_principal": "mae",
        "unidad": "horas",
        "test": {
            "modelo": {"nombre": selection["modelo"], **model_metrics},
            "baseline": {"nombre": "Baseline (mediana)", **baseline_metrics},
            "reduccion_mae_vs_baseline_pct": 100 * (1 - model_metrics["mae"] / baseline_metrics["mae"]),
        },
        "validacion": selection["candidatos"],
    }

    estimator = model.named_steps["regressor"]
    model_params = {key: value for key, value in estimator.get_params().items()
                    if value is None or isinstance(value, (bool, int, float, str))}
    metadata = {
        "version": args.version,
        "fecha_creacion": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tipo_problema": "regresion",
        "variable_objetivo": TARGET,
        "unidad_objetivo": "horas",
        "algoritmo": type(estimator).__name__,
        "modelo_seleccionado": selection["modelo"],
        "usa_pasajeros": selection["usa_pasajeros"],
        "variables_entrada": [{"nombre": name, "tipo": FEATURE_TYPES[name]} for name in PREDICTORS],
        "tratamiento_entrada": {
            "nulos_categoricas": "se reemplazan por 'DESCONOCIDA' dentro del pipeline",
            "nulos_numericas": "mediana del entrenamiento (SimpleImputer dentro del pipeline)",
            "categorias_no_vistas": "Random Forest: código -1; regresión lineal: se ignoran en el one-hot",
            "columnas_adicionales": "se descartan (remainder='drop')",
        },
        "parametros_modelo": model_params,
        "parametros_random_forest_solicitados": rf_params,
        "semilla": args.seed,
        "datos": {
            "fuente": DATA_SOURCE,
            "archivo": Path(args.data).name,
            "sha256": file_sha256(args.data),
            "periodo_salidas": list(VALID_YEARS),
            "exclusiones": exclusions,
            "filas_usadas": len(data),
            "muestra_max_rows": args.max_rows,
        },
        "separacion": {
            "estrategia": "aleatoria train/test; selección con validación interna derivada solo de train; "
                          "reajuste con todo train y una única evaluación en test",
            "test_size": args.test_size,
            "val_size": args.val_size,
            "filas_train": len(train),
            "filas_test": len(test),
            "filas_train_interno": selection["filas_train_interno"],
            "filas_validacion": selection["filas_validacion"],
            "regla_pasajeros": f"se incluye si mejora la MAE de validación al menos {PASSENGERS_MIN_IMPROVEMENT:.0%}",
        },
        "metrica_evaluacion": "MAE (principal), RMSE y R², en horas",
        "entorno": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "dependencias": "fase-2/requirements.txt",
        },
        "codigo": {"commit": git_commit()},
    }

    paths = save_version(args.output_dir, args.version, model, metrics, metadata)

    print(f"\nVersión {args.version} guardada en {time.time() - started:.0f} s")
    print(f"  Modelo   ({selection['modelo']}): MAE {model_metrics['mae']:.4f} h | "
          f"RMSE {model_metrics['rmse']:.4f} h | R² {model_metrics['r2']:.4f}")
    print(f"  Baseline (mediana): MAE {baseline_metrics['mae']:.4f} h | "
          f"RMSE {baseline_metrics['rmse']:.4f} h | R² {baseline_metrics['r2']:.4f}")
    for path in paths.values():
        print(f"  {path}")


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (DataValidationError, VersionExistsError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
