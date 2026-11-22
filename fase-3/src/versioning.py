"""Almacenamiento y carga de versiones del modelo en carpetas models/<version>/.

Cada versión contiene modelo.joblib (pipeline completo), metrics.json y metadata.json.
Una versión existente nunca se sobrescribe.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import joblib

MODEL_FILE = "modelo.joblib"
METRICS_FILE = "metrics.json"
METADATA_FILE = "metadata.json"
VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class VersionExistsError(FileExistsError):
    """Se intentó crear una versión con un identificador que ya existe."""


def version_dir(output_dir, version):
    """Ruta de la versión, validando que el identificador sea un nombre de carpeta simple."""
    if not VERSION_PATTERN.match(version):
        raise ValueError(f"Identificador de versión inválido: {version!r} (use letras, números, '.', '_' o '-')")
    return Path(output_dir) / version


def ensure_version_available(output_dir, version):
    """Falla antes de entrenar si la versión ya existe."""
    path = version_dir(output_dir, version)
    if path.exists():
        raise VersionExistsError(f"La versión '{version}' ya existe en {path}; use otro identificador con --version")
    return path


def save_version(output_dir, version, model, metrics, metadata):
    """Guarda los tres archivos en una carpeta temporal y la renombra al final.

    Así una ejecución interrumpida no deja una versión incompleta con el nombre definitivo.
    """
    target = ensure_version_available(output_dir, version)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=target.parent))
    try:
        joblib.dump(model, staging / MODEL_FILE, compress=3)
        for name, content in ((METRICS_FILE, metrics), (METADATA_FILE, metadata)):
            with open(staging / name, "w", encoding="utf-8") as file:
                json.dump(content, file, ensure_ascii=False, indent=2)
        ensure_version_available(output_dir, version)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {name: target / name for name in (MODEL_FILE, METRICS_FILE, METADATA_FILE)}


def load_version(path):
    """Carga una versión a partir de su carpeta o de su archivo modelo.joblib.

    Devuelve (modelo, metadata); metadata es None si la versión no tiene metadata.json.
    """
    path = Path(path)
    model_path = path / MODEL_FILE if path.is_dir() else path
    if not model_path.is_file():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")
    model = joblib.load(model_path)

    metadata_path = model_path.parent / METADATA_FILE
    metadata = None
    if metadata_path.is_file():
        with open(metadata_path, encoding="utf-8") as file:
            metadata = json.load(file)
    return model, metadata


def git_commit():
    """Hash del commit del código; dentro de Docker se toma de la variable GIT_COMMIT."""
    if os.environ.get("GIT_COMMIT"):
        return {"hash": os.environ["GIT_COMMIT"], "cambios_sin_commit": None}
    cwd = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                                text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=cwd.parent,
                                capture_output=True, text=True, check=True).stdout.strip()
        return {"hash": commit, "cambios_sin_commit": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"hash": "desconocido", "cambios_sin_commit": None}
