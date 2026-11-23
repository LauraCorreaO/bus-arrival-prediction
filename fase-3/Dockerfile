# Imagen de la Fase 3: API REST (FastAPI + uvicorn). Parte de la imagen de la Fase 2.
# Misma versión de Python registrada en models/*/metadata.json (3.13.5).
FROM python:3.13.5-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Código. Los modelos, los datos de entrenamiento y el registro se montan desde el anfitrión:
#   -v "$PWD/models:/app/models"      versiones (lectura para /predict, escritura para /train)
#   -v "$PWD/data:/app/data"          datos crudos para /train (datos_completos.csv.gz)
#   -v "$PWD/registro:/app/registro"  registro de /predict (fase 4), persiste entre ejecuciones
COPY src/ src/
COPY tests/ tests/
COPY api.py train.py predict.py cliente.py ./
COPY data/ejemplo_prediccion.csv data/

ARG GIT_COMMIT=""
ENV GIT_COMMIT=${GIT_COMMIT} \
    MODELO_VERSION=v1 \
    MODELS_DIR=/app/models \
    REGISTRO_PATH=/app/registro/solicitudes.csv

VOLUME ["/app/models", "/app/registro"]
EXPOSE 8000

# No se entrena durante el build: la API solo carga la versión indicada en MODELO_VERSION.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
