# Predicción de Duración de Viaje — Terminal de Transporte

Proyecto integrador del curso **Modelos y Simulación de Sistemas I**. Consiste en llevar un modelo de Machine Learning desde un notebook hasta un prototipo desplegable, en cuatro fases acumulativas: modelo predictivo, scripts y Docker, API REST, y monitoreo básico.

## Integrantes
- Laura Correa Ochoa
- Miguel Cerquera Arias
- Santiago David López

## Problema

Predecir la **duración de un viaje** (en horas) entre la salida y la llegada a la Terminal de Transporte, a partir de información disponible al momento de la salida (ruta, empresa, terminal, clase de vehículo, horario, número de pasajeros). Es un problema de **regresión** (no series de tiempo): cada viaje es una observación independiente.

## Dataset

- **Fuente:** [Datos Abiertos Colombia — Llegadas y Salidas de Vehículos de la Terminal de Transporte](https://www.datos.gov.co/Transporte/Llegada-y-Salidas-de-Veh-culos-de-la-Terminal-de-T/pfsr-mdyi/about_data) (recurso `pfsr-mdyi`)
- **Acceso:** API SoQL (`https://www.datos.gov.co/resource/pfsr-mdyi.json`), sin necesidad de descarga manual en CSV.
- **Alcance:** todas las clases de vehículo (BUS, MICROBUS, CAMIONETA, AUTOMOVIL, DUO BUS), conjunto completo: 2.617.130 registros (2023–2025).
- **Copia en el repositorio:** los datos crudos y los conjuntos de entrenamiento y prueba se incluyen comprimidos (`.csv.gz`) en `fase-1/data/`; ver [`fase-1/README.md`](fase-1/README.md) para cómo leerlos.

## Estructura del repositorio

```
fase-1/           Modelo predictivo (notebooks, datos, modelo entrenado)
fase-2/           Scripts (train.py / predict.py) y Docker
fase-3/           API REST
fase-4/           Monitoreo básico
```

## Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter notebook fase-1/notebooks/
```

Los notebooks de `fase-1/` se ejecutan en orden y de principio a fin; el detalle está en [`fase-1/README.md`](fase-1/README.md).

## Flujo de trabajo en GitHub

- Rama `develop` para integración, ramas `feature/*` por integrante/tarea, Pull Requests hacia `develop`, y `main` solo para las entregas.
- Este README se mantiene actualizado en cada fase.
