"""Cliente de prueba de la API: consulta /health y envía viajes reales a /predict.

Uso (con la API en ejecución):
    python cliente.py                         # API en http://localhost:8000
    python cliente.py --url http://otro:8000  # o API_URL=...
    python cliente.py --sin-campo empresa     # envía además un registro sin 'empresa' (debe dar 422)

Los registros salen de data/ejemplo_prediccion.csv, con los nombres y tipos de
models/v1/metadata.json (variables_entrada): texto para las categóricas, número para las
numéricas y null para un dato faltante. Solo usa la biblioteca estándar.
"""

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EJEMPLOS = BASE_DIR / "data" / "ejemplo_prediccion.csv"
for _stream in (sys.stdout, sys.stderr):  # tildes legibles en la consola de Windows
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

TIPOS = {"nombre_sucursal": "string", "empresa": "string", "ruta_origen": "string", "subregi_n": "string",
         "clase_veh_culo": "string", "pasajeros": "float", "hora_salida_decimal": "float",
         "dia_semana": "int", "mes": "int"}


def convertir(fila):
    """Fila del CSV (texto) -> objeto JSON con los tipos usados al entrenar."""
    viaje = {}
    for nombre, tipo in TIPOS.items():
        valor = fila.get(nombre, "")
        if valor == "":
            viaje[nombre] = None
        elif tipo == "int":
            viaje[nombre] = int(float(valor))
        elif tipo == "float":
            viaje[nombre] = float(valor)
        else:
            viaje[nombre] = valor
    return viaje


def enviar(url, metodo="GET", cuerpo=None):
    datos = json.dumps(cuerpo).encode("utf-8") if cuerpo is not None else None
    solicitud = urllib.request.Request(url, data=datos, method=metodo,
                                       headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(solicitud, timeout=60) as respuesta:
            return respuesta.status, json.loads(respuesta.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def mostrar(titulo, estado, cuerpo, solicitud=None):
    print(f"\n== {titulo} -> HTTP {estado}")
    if solicitud is not None:
        print("Solicitud:", json.dumps(solicitud, ensure_ascii=False))
    print("Respuesta:", json.dumps(cuerpo, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cliente de prueba de la API de duración de viajes")
    parser.add_argument("--url", default=os.environ.get("API_URL", "http://localhost:8000"))
    parser.add_argument("-n", type=int, default=2, help="Número de registros válidos a enviar (por defecto 2)")
    parser.add_argument("--sin-campo", default=None, help="Envía también un registro sin este campo obligatorio")
    args = parser.parse_args(argv)
    url = args.url.rstrip("/")

    try:
        mostrar("GET /health", *enviar(f"{url}/health"))
    except urllib.error.URLError as error:
        print(f"No se pudo conectar con {url}: {error.reason}", file=sys.stderr)
        return 1

    with open(EJEMPLOS, newline="", encoding="utf-8") as file:
        viajes = [convertir(fila) for fila in csv.DictReader(file)][:args.n]
    for viaje in viajes:
        mostrar("POST /predict", *enviar(f"{url}/predict", "POST", viaje), solicitud=viaje)

    if args.sin_campo:
        incompleto = {k: v for k, v in viajes[0].items() if k != args.sin_campo}
        mostrar(f"POST /predict sin '{args.sin_campo}'", *enviar(f"{url}/predict", "POST", incompleto),
                solicitud=incompleto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
