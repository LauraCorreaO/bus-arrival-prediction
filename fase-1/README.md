# Fase 1 — Modelo predictivo

## Objetivo

Construir un modelo de Machine Learning capaz de predecir la **duración de un viaje de bus** (en horas) en la Terminal de Transporte, a partir de datos disponibles al momento de la salida.

## Estado actual

- [x] Selección del problema y verificación del dataset — [`docs/01-seleccion-del-problema.md`](docs/01-seleccion-del-problema.md)
- [ ] Exploración de datos (EDA)
- [ ] Preparación de datos (limpieza, manejo de nulos, evitar fuga de información)
- [ ] Entrenamiento y evaluación del modelo
- [ ] Guardado del modelo

## Estructura

```
data/
  raw/          Datos crudos descargados de la API (no versionados)
  processed/    Datos limpios/transformados (no versionados)
notebooks/      Notebooks ejecutables de exploración y modelado
models/         Modelo(s) entrenado(s) serializado(s)
docs/           Documentación de decisiones tomadas en esta fase
```

## Variable objetivo

`duracion_viaje_horas` = (`fecha_llegada` − `fecha_salida`) en horas. Ver justificación completa y por qué no es series de tiempo en [`docs/01-seleccion-del-problema.md`](docs/01-seleccion-del-problema.md).
