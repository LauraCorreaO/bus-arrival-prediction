from pathlib import Path

import sys

FASE2_DIR = Path(__file__).resolve().parent.parent
if str(FASE2_DIR) not in sys.path:
    sys.path.insert(0, str(FASE2_DIR))

import json

import pandas as pd
import pytest

# Importar las funciones de predict.py
from predict import validate_and_preprocess_input


# ----------------------------------------------------------------------
# FIXTURES (Datos sintéticos en memoria)
# ----------------------------------------------------------------------
@pytest.fixture
def sample_input_data():
    """Genera un DataFrame sintético con datos válidos sin variable objetivo."""
    return pd.DataFrame(
        {
            "nombre_sucursal": ["Sucursal Norte", "Sucursal Sur"],
            "empresa": ["Empresa A", "Empresa B"],
            "ruta_origen": ["Ruta 1", "Ruta 2"],
            "subregi_n": ["Subregion 1", "Subregion 2"],
            "clase_veh_culo": ["Bus", "Microbus"],
            "pasajeros": [25, 12],
            "hora_salida_decimal": [8.5, 14.0],
            "dia_semana": [1, 3],
            "mes": [3, 5],
        }
    )


# ----------------------------------------------------------------------
# SUITE DE PRUEBAS PARA LA PERSONA B
# ----------------------------------------------------------------------


# Test 1: Comprobar lectura de metadata
def test_metadata_structure():
    """Verifica que metadata.json de v1 exista y contenga información válida del modelo."""
    metadata_path = FASE2_DIR / "models" / "v1" / "metadata.json"
    assert metadata_path.exists(), "El archivo metadata.json v1 no existe."

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["version"] == "v1"
    assert meta["tipo_problema"] == "regresion"
    assert meta["variable_objetivo"] == "duracion_viaje_horas"
    assert meta["unidad_objetivo"] == "horas"
    assert meta["algoritmo"] == "RandomForestRegressor"
    assert isinstance(meta["usa_pasajeros"], bool)
    assert isinstance(meta["variables_entrada"], list)
    assert all(
        {"nombre", "tipo"} <= variable.keys()
        for variable in meta["variables_entrada"]
    )


# Test 2: Comprobar rechazo cuando faltan columnas obligatorias
def test_predict_missing_columns(sample_input_data):
    """Verifica que la función falle con SystemExit si falta una columna requerida."""
    df_incomplete = sample_input_data.drop(columns=["pasajeros"])
    expected_features = list(sample_input_data.columns)

    with pytest.raises(SystemExit) as exc_info:
        validate_and_preprocess_input(df_incomplete, expected_features)

    assert exc_info.value.code == 2


# Test 3: Manejo de columnas sobrantes o extras
def test_predict_extra_columns(sample_input_data):
    """Verifica que las columnas extra no esperadas sean filtradas."""
    df_extra = sample_input_data.copy()
    df_extra["columna_irrelevante"] = [99, 100]

    expected_features = list(sample_input_data.columns)
    df_clean = validate_and_preprocess_input(df_extra, expected_features)

    assert "columna_irrelevante" not in df_clean.columns
    assert list(df_clean.columns) == expected_features


# Test 4: Prueba con tmp_path (Entorno aislado)
def test_predict_output_file_isolation(tmp_path, sample_input_data):
    """Predice con v1 en un directorio temporal: una predicción por fila y en el mismo orden."""
    import predict

    input_file = tmp_path / "test_input.csv"
    output_file = tmp_path / "salida" / "predicciones.csv"
    sample_input_data.to_csv(input_file, index=False)
    assert input_file.exists()

    code = predict.main(["--model", str(FASE2_DIR / "models" / "v1" / "modelo.joblib"),
                         "--input", str(input_file), "--output", str(output_file)])
    assert code == 0
    output = pd.read_csv(output_file)
    assert len(output) == len(sample_input_data)
    assert list(output["mes"]) == list(sample_input_data["mes"])
    assert output["prediccion_duracion_horas"].notna().all()


# Test 5: PRUEBA OBLIGATORIA CONTRA DATA LEAKAGE
def test_no_data_leakage_from_arrival_columns():
    """Valida programáticamente que ninguna columna de la metadata derive de la hora de llegada."""
    metadata_path = FASE2_DIR / "models" / "v1" / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    features = [variable["nombre"] for variable in meta["variables_entrada"]]

    forbidden_terms = ["llegada", "arrival", "fecha_llegada", "hora_llegada"]

    for feat in features:
        for forbidden in forbidden_terms:
            assert (
                forbidden not in str(feat).lower()
            ), f"DATA LEAKAGE DETECTADO: La característica '{feat}' contiene el término prohibido '{forbidden}'"