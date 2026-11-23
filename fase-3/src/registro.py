"""Registro de solicitudes a /predict en un CSV (fase 4).

Una fila por intento: fecha_hora,estado_http,version,prediccion. El archivo se abre en modo
agregar y el encabezado se escribe solo al crearlo, así el historial se conserva entre
ejecuciones. La ruta se toma de REGISTRO_PATH (por defecto registro/solicitudes.csv junto a
api.py); en Docker se monta esa carpeta desde el anfitrión.
"""

import csv
import os
import threading
from datetime import datetime
from pathlib import Path

ENCABEZADO = ["fecha_hora", "estado_http", "version", "prediccion"]
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "registro" / "solicitudes.csv"
_lock = threading.Lock()


def registro_path():
    return Path(os.environ.get("REGISTRO_PATH", DEFAULT_PATH))


def registrar(estado_http, version=None, prediccion=None, path=None):
    """Agrega una fila. version y prediccion quedan vacías si no se conocen."""
    path = Path(path) if path else registro_path()
    fila = [datetime.now().isoformat(timespec="seconds"), int(estado_http),
            version or "", "" if prediccion is None else prediccion]
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        nuevo = not path.is_file() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if nuevo:
                writer.writerow(ENCABEZADO)
            writer.writerow(fila)
