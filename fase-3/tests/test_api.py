"""Pruebas de la API (fase 3) con el cliente de pruebas de FastAPI.

Se usa un modelo pequeño entrenado en memoria con datos sintéticos (unos segundos), guardado en
una carpeta temporal; no se carga v1 ni se ejecuta un entrenamiento completo. El registro de
solicitudes también va a una carpeta temporal.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FASE3_DIR = Path(__file__).resolve().parent.parent
if str(FASE3_DIR) not in sys.path:
    sys.path.insert(0, str(FASE3_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from src.config import FEATURE_TYPES, PREDICTORS  # noqa: E402
from src.pipeline import build_model  # noqa: E402
from src.versioning import save_version  # noqa: E402

VERSION_PRUEBA = "v-prueba"
VIAJE_VALIDO = {
    "nombre_sucursal": "TERMINAL DEL SUR",
    "empresa": "TRANSPORTES JERICO",
    "ruta_origen": "JERICO",
    "subregi_n": "SUR",
    "clase_veh_culo": "AUTOMOVIL",
    "pasajeros": 1,
    "hora_salida_decimal": 14.0,
    "dia_semana": 6,
    "mes": 7,
}


@pytest.fixture(scope="session")
def modelos_dir(tmp_path_factory):
    """Guarda una versión diminuta (Random Forest de 5 árboles, 300 filas sintéticas)."""
    rng = np.random.default_rng(0)
    n = 300
    data = pd.DataFrame({
        "nombre_sucursal": rng.choice(["TERMINAL DEL SUR", "TERMINAL DEL NORTE"], n),
        "empresa": rng.choice(["TRANSPORTES JERICO", "SOTRAGUR S.A"], n),
        "ruta_origen": rng.choice(["JERICO", "GUARNE", "COCORNA"], n),
        "subregi_n": rng.choice(["SUR", "ORIENTE"], n),
        "clase_veh_culo": rng.choice(["BUS", "AUTOMOVIL"], n),
        "pasajeros": rng.integers(1, 40, n).astype(float),
        "hora_salida_decimal": rng.uniform(0, 23.9, n),
        "dia_semana": rng.integers(0, 7, n),
        "mes": rng.integers(1, 13, n),
    })
    target = 1 + data["hora_salida_decimal"] / 10 + rng.normal(0, 0.1, n)
    model = build_model("random_forest", False, seed=0,
                        rf_params={"n_estimators": 5, "min_samples_leaf": 5, "max_samples": 0.5})
    model.fit(data[PREDICTORS], target)

    directory = tmp_path_factory.mktemp("models")
    metadata = {"version": VERSION_PRUEBA, "tipo_problema": "regresion",
                "variables_entrada": [{"nombre": c, "tipo": FEATURE_TYPES[c]} for c in PREDICTORS]}
    save_version(directory, VERSION_PRUEBA, model, {"version": VERSION_PRUEBA}, metadata)
    return directory


@pytest.fixture
def client(modelos_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(modelos_dir))
    monkeypatch.setenv("MODELO_VERSION", VERSION_PRUEBA)
    monkeypatch.setenv("REGISTRO_PATH", str(tmp_path / "solicitudes.csv"))
    return TestClient(api.app)


def filas_registro(tmp_path):
    with open(tmp_path / "solicitudes.csv", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_health_responde_200_y_estado(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"estado": "en ejecución"}


def test_entrada_valida_devuelve_prediccion_y_version(client, tmp_path):
    response = client.post("/predict", json=VIAJE_VALIDO)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediccion", "version"}
    assert isinstance(body["prediccion"], float)
    assert body["version"] == VERSION_PRUEBA

    [fila] = filas_registro(tmp_path)
    assert fila["estado_http"] == "200" and fila["version"] == VERSION_PRUEBA and fila["prediccion"]


def test_campo_obligatorio_ausente_se_rechaza(client, tmp_path):
    viaje = {k: v for k, v in VIAJE_VALIDO.items() if k != "empresa"}
    response = client.post("/predict", json=viaje)
    assert response.status_code == 422
    body = response.json()
    assert "empresa" in body["mensaje"]
    assert body["detalles"][0]["campo"] == "empresa"

    # El rechazo de validación también queda registrado, sin versión ni predicción
    [fila] = filas_registro(tmp_path)
    assert (fila["estado_http"], fila["version"], fila["prediccion"]) == ("422", "", "")
