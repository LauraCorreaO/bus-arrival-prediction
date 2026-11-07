# Imagen de la Fase 2: permite ejecutar train.py y predict.py.
# Misma versión de Python registrada en models/*/metadata.json (3.13.5).
FROM python:3.13.5-slim

# Sin archivos .pyc, salida sin búfer y UTF-8 para los mensajes con tildes
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencias primero: esta capa se reutiliza mientras requirements.txt no cambie
COPY requirements.txt .
RUN pip install -r requirements.txt

# Código de la aplicación (los datos, modelos y salidas se montan desde el anfitrión)
COPY src/ src/
COPY tests/ tests/
COPY train.py predict.py ./
COPY data/ejemplo_prediccion.csv data/

# Commit del código; se pasa en el build o al ejecutar (-e GIT_COMMIT=...) para metadata.json
ARG GIT_COMMIT=""
ENV GIT_COMMIT=${GIT_COMMIT}

# Sin ENTRYPOINT: el comando se indica al ejecutar (python train.py ... o python predict.py ...).
# No se entrena durante el build.
CMD ["python", "predict.py", "--help"]
