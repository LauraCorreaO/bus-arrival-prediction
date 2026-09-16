# Fase 1 — Modelo predictivo

## Objetivo

Construir un modelo de Machine Learning capaz de predecir la **duración de un viaje** (en horas) en la Terminal de Transporte, a partir de datos disponibles al momento de la salida.

## Estado actual

- [x] Selección del problema y verificación del dataset
- [x] Exploración de datos (EDA) — `notebooks/01-exploracion-datos.ipynb`
- [x] Preparación de datos (limpieza, manejo de nulos, evitar fuga de información) — `notebooks/02-preparacion-datos.ipynb`
- [ ] Entrenamiento y evaluación del modelo
- [ ] Guardado del modelo

## Estructura

```
data/
  raw/          Datos crudos descargados de la API (no versionados, se regeneran con el notebook 1)
  processed/    Datos limpios/transformados (no versionados, se regeneran con el notebook 2)
notebooks/      Notebooks ejecutables de exploración y modelado
models/         Modelo(s) entrenado(s) serializado(s)
docs/           Notas de trabajo personales (no versionadas, no forman parte del repositorio)
```

## Variable objetivo

`duracion_viaje_horas`, calculada con `hora_de_llegada` y `fecha_hora_salida_origen` (con respaldo en `fecha_salida` cuando falta) — ver el notebook de exploración (Paso 6) para la justificación completa de por qué no se usa directamente `fecha_llegada - fecha_salida`.
