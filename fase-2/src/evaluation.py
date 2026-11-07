"""Métricas y selección del modelo con una validación interna derivada de train.

El conjunto de prueba no interviene aquí: la selección se cierra antes de evaluarlo.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import PASSENGERS_MIN_IMPROVEMENT, PREDICTORS, TARGET
from src.pipeline import ALGORITHM_NAMES, build_model


def compute_metrics(y_true, y_pred):
    """MAE (métrica principal), RMSE y R², en horas."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def candidate_name(algorithm, use_passengers):
    if algorithm == "baseline":
        return ALGORITHM_NAMES[algorithm]
    return f"{ALGORITHM_NAMES[algorithm]} ({'con' if use_passengers else 'sin'} pasajeros)"


def select_model(train, val_size, seed, rf_params=None):
    """Elige algoritmo y uso de pasajeros con la MAE de una validación interna.

    Para cada algoritmo, pasajeros se conserva solo si mejora la MAE de validación al
    menos PASSENGERS_MIN_IMPROVEMENT. Entre el baseline y los candidatos resultantes se
    elige la menor MAE de validación.
    """
    inner, validation = train_test_split(train, test_size=val_size, random_state=seed)
    X_inner, y_inner = inner[PREDICTORS], inner[TARGET]
    X_val, y_val = validation[PREDICTORS], validation[TARGET]

    def evaluate(algorithm, use_passengers):
        model = build_model(algorithm, use_passengers, seed=seed, rf_params=rf_params)
        model.fit(X_inner, y_inner)
        metrics = compute_metrics(y_val, model.predict(X_val))
        print(f"  {candidate_name(algorithm, use_passengers)}: MAE validación {metrics['mae']:.4f} h")
        return {"modelo": candidate_name(algorithm, use_passengers), "algoritmo": algorithm,
                "usa_pasajeros": use_passengers, **metrics}

    rows = [evaluate("baseline", False)]
    finalists = [rows[0]]
    for algorithm in ("linear", "random_forest"):
        with_passengers, without_passengers = evaluate(algorithm, True), evaluate(algorithm, False)
        rows += [with_passengers, without_passengers]
        improvement = (without_passengers["mae"] - with_passengers["mae"]) / without_passengers["mae"]
        finalists.append(with_passengers if improvement >= PASSENGERS_MIN_IMPROVEMENT else without_passengers)

    selected = min(finalists, key=lambda row: row["mae"])
    return {
        "algoritmo": selected["algoritmo"],
        "usa_pasajeros": selected["usa_pasajeros"],
        "modelo": selected["modelo"],
        "filas_train_interno": len(inner),
        "filas_validacion": len(validation),
        "candidatos": rows,
    }
