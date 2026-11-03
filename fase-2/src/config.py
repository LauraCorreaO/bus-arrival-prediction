"""Configuración compartida: esquema de datos y valores por defecto.

Este módulo solo define constantes; no lee archivos ni entrena nada al importarse.
"""

# Variable objetivo: horas transcurridas entre la salida y la llegada.
TARGET = "duracion_viaje_horas"

# Predictoras: solo información conocida al momento de la salida.
CATEGORICAL = ["nombre_sucursal", "empresa", "ruta_origen", "subregi_n", "clase_veh_culo"]
NUMERIC = ["pasajeros", "hora_salida_decimal", "dia_semana", "mes"]
PREDICTORS = CATEGORICAL + NUMERIC

# Tipos esperados de cada predictora en el CSV de entrada (se guardan en metadata.json).
FEATURE_TYPES = {
    "nombre_sucursal": "string",
    "empresa": "string",
    "ruta_origen": "string",
    "subregi_n": "string",
    "clase_veh_culo": "string",
    "pasajeros": "float",           # admite nulos: se imputa dentro del pipeline
    "hora_salida_decimal": "float",  # hora decimal en [0, 24)
    "dia_semana": "int",             # 0 = lunes ... 6 = domingo
    "mes": "int",                    # 1 ... 12
}

# Columnas de la salida y de la llegada en los datos crudos.
DEPARTURE_COLUMNS = ["fecha_hora_salida_origen", "fecha_salida"]
ARRIVAL_COLUMNS = ["fecha_llegada", "hora_de_llegada"]

# Columnas que deben existir en el CSV crudo para entrenar.
RAW_REQUIRED_COLUMNS = CATEGORICAL + ["pasajeros"] + DEPARTURE_COLUMNS + ["hora_de_llegada"]

# Columnas que describen la llegada o el estado del viaje: nunca pueden ser predictoras
# ni usarse para derivarlas (fuga de información).
LEAKAGE_COLUMNS = ARRIVAL_COLUMNS + ["estado", "ruta_destino"]

# Período publicado del recurso; se filtra usando solo la fecha de salida.
VALID_YEARS = (2023, 2025)

# Partición y selección (iguales a los notebooks 02 y 05 de la Fase 1).
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.20
DEFAULT_VAL_SIZE = 0.20
PASSENGERS_MIN_IMPROVEMENT = 0.01   # pasajeros se conserva solo si mejora >= 1 % la MAE de validación

# Hiperparámetros por defecto del Random Forest (modelo oficial de la Fase 1).
DEFAULT_RF_PARAMS = {
    "n_estimators": 80,
    "min_samples_leaf": 50,
    "max_samples": 0.5,
}
