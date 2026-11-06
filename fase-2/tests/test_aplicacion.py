"""Pruebas de la aplicación: entrenamiento, carga, nulos, entrada inválida y versionamiento.

Usan datos sintéticos pequeños con el esquema crudo de datos.gov.co y directorios temporales
(tmp_path), así que no descargan nada ni modifican los modelos de la entrega.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FASE2_DIR = Path(__file__).resolve().parent.parent
if str(FASE2_DIR) not in sys.path:
    sys.path.insert(0, str(FASE2_DIR))

import predict  
import train 
from src.config import PREDICTORS, TARGET  
from predict import PREDICTION_COLUMN, validate_and_preprocess_input  
from src.versioning import load_version 

N_ROWS = 300
FAST_RF = ["--n-estimators", "5", "--min-samples-leaf", "5"]


def run_predict(argv):
    """Ejecuta predict.main y devuelve el código de salida (predict usa sys.exit en los errores)."""
    try:
        return predict.main(argv)
    except SystemExit as exit_:
        return exit_.code


def schema(metadata):
    types = {v["nombre"]: v["tipo"] for v in metadata["variables_entrada"]}
    return list(types), types


def make_raw_data(n=N_ROWS, seed=0):
    """CSV crudo sintético: la duración depende de la ruta y de la hora de salida."""
    rng = np.random.default_rng(seed)
    routes = {"RIONEGRO": 1.0, "JERICO": 3.0, "BOGOTA-MEDELLIN": 9.0}
    route = rng.choice(list(routes), n)
    departure = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n), unit="min")
    hours = np.array([routes[r] for r in route]) + rng.normal(0, 0.2, n) + departure.hour / 24
    arrival = departure + pd.to_timedelta(hours, unit="h")
    fmt = "%Y-%m-%dT%H:%M:%S.000"
    return pd.DataFrame({
        "nombre_sucursal": rng.choice(["Terminal del Sur", "Terminal del Norte"], n),
        "empresa": rng.choice(["EMPRESA A", "EMPRESA B"], n),
        "ruta_origen": route,
        "subregi_n": rng.choice(["Sur", "Oriente"], n),
        "clase_veh_culo": rng.choice(["BUS", "MICROBUS"], n),
        "pasajeros": rng.integers(1, 40, n),
        "fecha_hora_salida_origen": departure.strftime(fmt),
        "fecha_salida": departure.strftime(fmt),
        "fecha_llegada": arrival.strftime(fmt),
        "hora_de_llegada": arrival.strftime(fmt),
    })


def new_rows():
    return pd.DataFrame({
        "nombre_sucursal": ["Terminal del Sur", "Terminal del Norte", "Terminal del Sur"],
        "empresa": ["EMPRESA A", "EMPRESA B", "EMPRESA A"],
        "ruta_origen": ["JERICO", "RIONEGRO", "BOGOTA-MEDELLIN"],
        "subregi_n": ["Sur", "Oriente", "Sur"],
        "clase_veh_culo": ["BUS", "MICROBUS", "BUS"],
        "pasajeros": [10, 20, 30],
        "hora_salida_decimal": [8.5, 14.0, 22.25],
        "dia_semana": [0, 3, 6],
        "mes": [1, 6, 12],
    })


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Entrena una única versión v1 con datos sintéticos y la reutiliza en varias pruebas."""
    base = tmp_path_factory.mktemp("app")
    data = base / "datos.csv"
    make_raw_data().to_csv(data, index=False)
    models = base / "models"
    code = train.main(["--data", str(data), "--output-dir", str(models), "--version", "v1", "--seed", "42", *FAST_RF])
    assert code == 0
    return {"base": base, "data": data, "models": models}


def test_entrenamiento_y_almacenamiento(trained):
    """El entrenamiento genera un artefacto cargable, métricas (modelo y baseline) y metadatos."""
    version = trained["models"] / "v1"
    assert {"modelo.joblib", "metrics.json", "metadata.json"} <= {p.name for p in version.iterdir()}

    model, metadata = load_version(version / "modelo.joblib")
    assert hasattr(model, "predict")
    assert metadata["version"] == "v1"
    assert metadata["variable_objetivo"] == TARGET
    assert [v["nombre"] for v in metadata["variables_entrada"]] == PREDICTORS
    assert metadata["semilla"] == 42
    assert metadata["separacion"]["filas_train"] + metadata["separacion"]["filas_test"] == N_ROWS

    metrics = json.loads((version / "metrics.json").read_text(encoding="utf-8"))
    for key in ("modelo", "baseline"):
        assert {"mae", "rmse", "r2"} <= metrics["test"][key].keys()

    assert metrics["test"]["modelo"]["mae"] < metrics["test"]["baseline"]["mae"]


def test_carga_y_prediccion_conserva_resultados(trained, tmp_path):
    """El artefacto guardado da una predicción por fila, en orden e igual a la del modelo en memoria."""
    from src.data import load_data, prepare_data, split_data
    from src.pipeline import build_model

    model, metadata = load_version(trained["models"] / "v1")
    data, _ = prepare_data(load_data(trained["data"]))
    train_df, _ = split_data(data, 0.2, 42)
    algorithm = {"RandomForestRegressor": "random_forest", "LinearRegression": "linear",
                 "DummyRegressor": "baseline"}[metadata["algoritmo"]]
    in_memory = build_model(algorithm, metadata["usa_pasajeros"], seed=42,
                            rf_params={"n_estimators": 5, "min_samples_leaf": 5, "max_samples": 0.5})
    in_memory.fit(train_df[PREDICTORS], train_df[TARGET])
    np.testing.assert_allclose(model.predict(new_rows()), in_memory.predict(new_rows()), rtol=1e-9)

    rows = new_rows()[list(reversed(PREDICTORS))]
    rows.insert(0, "id_viaje", ["a", "b", "c"])
    input_csv, output_csv = tmp_path / "entrada.csv", tmp_path / "salida.csv"
    rows.to_csv(input_csv, index=False)
    code = run_predict(["--model", str(trained["models"] / "v1" / "modelo.joblib"),
                         "--input", str(input_csv), "--output", str(output_csv)])
    assert code == 0
    output = pd.read_csv(output_csv)
    assert len(output) == len(rows)
    assert "id_viaje" not in output.columns  # las columnas adicionales se descartan
    assert list(output["mes"]) == list(new_rows()["mes"])  # se conserva el orden de las filas
    np.testing.assert_allclose(output[PREDICTION_COLUMN], model.predict(new_rows()), rtol=1e-9)


def test_valores_faltantes_se_imputan(trained):
    """Celdas nulas en variables que los admiten se procesan sin eliminar filas."""
    model, metadata = load_version(trained["models"] / "v1")
    rows = new_rows().astype({"pasajeros": float, "hora_salida_decimal": float})
    rows.loc[0, "pasajeros"] = np.nan
    rows.loc[1, "empresa"] = None
    rows.loc[2, "hora_salida_decimal"] = np.nan
    prepared = validate_and_preprocess_input(rows, *schema(metadata))
    predictions = model.predict(prepared)
    assert len(predictions) == len(rows)
    assert not np.isnan(predictions).any()


@pytest.mark.parametrize("change, message", [
    (lambda df: df.drop(columns=["ruta_origen"]), "faltan las siguientes columnas"),
    (lambda df: df.assign(hora_salida_decimal=["8.5", "tarde", "1"]), "debe ser numérica"),
    (lambda df: df.assign(mes=[1, 13, 2]), "fuera de rango"),
])
def test_entrada_invalida_error_claro(trained, tmp_path, capsys, change, message):
    """Columna ausente o tipo/rango inválido: error comprensible, código != 0 y sin CSV de salida."""
    model, metadata = load_version(trained["models"] / "v1")
    bad = change(new_rows())
    with pytest.raises(SystemExit) as exc_info:
        validate_and_preprocess_input(bad, *schema(metadata))
    assert exc_info.value.code != 0
    assert message in capsys.readouterr().err

    input_csv, output_csv = tmp_path / "mala.csv", tmp_path / "salida.csv"
    bad.to_csv(input_csv, index=False)
    code = run_predict(["--model-version", "v1", "--models-dir", str(trained["models"]),
                         "--input", str(input_csv), "--output", str(output_csv)])
    assert code != 0
    assert message in capsys.readouterr().err
    assert not output_csv.exists()


def test_modelo_inexistente_error_claro(tmp_path, capsys):
    input_csv = tmp_path / "entrada.csv"
    new_rows().to_csv(input_csv, index=False)
    code = run_predict(["--model", str(tmp_path / "no_existe" / "modelo.joblib"),
                         "--input", str(input_csv), "--output", str(tmp_path / "salida.csv")])
    assert code != 0
    assert "No se encontró" in capsys.readouterr().err


def test_versionamiento_no_sobrescribe(trained):
    """Una segunda versión conserva la primera, ambas se cargan, y un id repetido no sobrescribe."""
    models = trained["models"]
    v1_model = models / "v1" / "modelo.joblib"
    before = (v1_model.stat().st_mtime_ns, v1_model.read_bytes())

    args = ["--data", str(trained["data"]), "--output-dir", str(models), "--seed", "7", *FAST_RF]
    assert train.main([*args, "--version", "v2"]) == 0
    assert train.main([*args, "--version", "v1"]) != 0  # identificador repetido: se rechaza

    assert (v1_model.stat().st_mtime_ns, v1_model.read_bytes()) == before
    for version in ("v1", "v2"):
        model, metadata = load_version(models / version)
        assert metadata["version"] == version
        assert len(model.predict(new_rows())) == len(new_rows())


def test_predictoras_sin_informacion_de_llegada(trained):
    """Ninguna predictora guardada deriva de la llegada (fuga de información)."""
    _, metadata = load_version(trained["models"] / "v1")
    for variable in metadata["variables_entrada"]:
        assert "llegada" not in variable["nombre"].lower()
