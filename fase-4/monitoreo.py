"""Métricas de operación de la API a partir del registro de solicitudes a /predict.

Uso (desde la raíz del repositorio):
    python fase-4/monitoreo.py                       # usa fase-3/registro/solicitudes.csv
    python fase-4/monitoreo.py ruta/al/registro.csv  # o REGISTRO_PATH=...

Muestra el total de solicitudes y el porcentaje de fallidas (códigos HTTP 4xx y 5xx).
Si el registro no existe o no tiene filas de datos, muestra total 0 y «sin solicitudes».
"""

import csv
import os
import sys
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "fase-3" / "registro" / "solicitudes.csv"


def leer_estados(path):
    """Códigos HTTP de cada fila de datos del registro (lista vacía si no existe)."""
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as file:
        return [int(fila["estado_http"]) for fila in csv.DictReader(file) if fila.get("estado_http")]


def calcular_metricas(estados):
    """Devuelve (total, fallidas, porcentaje); porcentaje es None si no hay solicitudes."""
    total = len(estados)
    fallidas = sum(1 for estado in estados if 400 <= estado <= 599)
    return total, fallidas, (100 * fallidas / total if total else None)


def formatear(total, fallidas, porcentaje):
    if porcentaje is None:
        return f"Total de solicitudes: {total}\nPorcentaje de fallos: sin solicitudes"
    texto = f"{porcentaje:.1f}".replace(".", ",")
    return (f"Total de solicitudes: {total}\n"
            f"Solicitudes fallidas (4xx/5xx): {fallidas}\n"
            f"Porcentaje de fallos: {texto} %")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else Path(os.environ.get("REGISTRO_PATH", DEFAULT_LOG))
    print(f"Registro: {path}")
    print(formatear(*calcular_metricas(leer_estados(path))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
