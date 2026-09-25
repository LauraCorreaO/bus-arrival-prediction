# Fase 1 — Predicción de la duración de un viaje en la Terminal de Transporte

## Integrantes
- Laura Correa Ochoa
- Miguel Cerquera Arias
- Santiago David López

## Descripción del problema

Predecir la **duración de un viaje** (en horas) entre su salida y su llegada a la Terminal de Transporte, usando únicamente información conocida al momento de la salida: ruta de origen, empresa, terminal, clase de vehículo, número de pasajeros, y hora, día de la semana y mes de salida. Es un problema de **regresión** (no series de tiempo): cada viaje es una observación independiente.

## Fuente de los datos

[Datos Abiertos Colombia — Llegadas y Salidas de Vehículos de la Terminal de Transporte](https://www.datos.gov.co/Transporte/Llegada-y-Salidas-de-Veh-culos-de-la-Terminal-de-T/pfsr-mdyi/about_data) (recurso `pfsr-mdyi`). Se descarga completo desde la API SoQL: 2.617.130 registros de 2023 a 2025, todas las clases de vehículo.

## Objetivo del modelo

Estimar el tiempo de viaje (ETA) de un vehículo que acaba de salir hacia la terminal, antes de que llegue.

## Algoritmo utilizado



## Métrica empleada



## Principales resultados



## Estado

- [x] Exploración de datos — `notebooks/01-exploracion-datos.ipynb`
- [x] Preparación de datos — `notebooks/02-preparacion-datos.ipynb`
- [ ] Modelo base y modelo predictivo
- [ ] Guardado del modelo

## Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter notebook fase-1/notebooks/
```

Ejecutar los notebooks **en orden y de principio a fin**: primero `01-exploracion-datos.ipynb` (descarga los datos completos a `data/raw/`; tarda unos minutos) y luego `02-preparacion-datos.ipynb` (genera `train.csv` y `test.csv` en `data/processed/`). Los archivos de datos no se versionan: se regeneran al ejecutar los notebooks.

## Estructura

```
data/raw/        Datos descargados de la API (no versionados)
data/processed/  Conjuntos de entrenamiento y prueba (no versionados)
notebooks/       Notebooks ejecutables
models/          Modelo entrenado
```
