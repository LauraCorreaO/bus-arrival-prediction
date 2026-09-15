# Predicción de Duración de Viaje — Terminal de Transporte

Proyecto integrador del curso **Modelos y Simulación de Sistemas I**. Consiste en llevar un modelo de Machine Learning desde un notebook hasta un prototipo desplegable, en cuatro fases acumulativas: modelo predictivo, scripts y Docker, API REST, y monitoreo básico.

## Problema

Predecir la **duración de un viaje de bus** (en horas) entre la salida y la llegada a la Terminal de Transporte, a partir de información disponible al momento de la salida (ruta, empresa, terminal, horario, número de pasajeros). Es un problema de **regresión** (no series de tiempo): cada viaje es una observación independiente.

Detalle completo de la selección del problema (variable objetivo, predictoras, justificación, verificación de requisitos del dataset): [`fase-1/docs/01-seleccion-del-problema.md`](fase-1/docs/01-seleccion-del-problema.md).

## Dataset

- **Fuente:** [Datos Abiertos Colombia — Llegadas y Salidas de Vehículos de la Terminal de Transporte](https://www.datos.gov.co/Transporte/Llegada-y-Salidas-de-Veh-culos-de-la-Terminal-de-T/pfsr-mdyi/about_data) (recurso `pfsr-mdyi`)
- **Acceso:** API SoQL (`https://www.datos.gov.co/resource/pfsr-mdyi.json`), sin necesidad de descarga manual en CSV.
- **Filtro aplicado:** `clase_vehiculo = BUS`.
- **Tamaño:** ~1.3 millones de registros en el recurso completo; se trabaja con una muestra manejable en un computador personal.

## Estructura del repositorio

```
fase-1/           Modelo predictivo (exploración, notebook, modelo entrenado)
  data/           Datos crudos y procesados (no versionados, ver .gitignore)
  notebooks/      Notebooks de exploración y entrenamiento
  models/         Modelo(s) entrenado(s) y serializado(s)
  docs/           Documentación de decisiones de la fase
fase-2/           Scripts (train.py / predict.py) y Docker
fase-3/           API REST
fase-4/           Monitoreo básico
```

## Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter notebook fase-1/notebooks/
```

## Equipo

_Pendiente completar con los integrantes del equipo._

## Flujo de trabajo en GitHub

- Rama `develop` para integración, ramas de `feature/` por integrante/tarea.
- Al menos un Pull Request por integrante.
- Este README se mantiene actualizado en cada fase.
