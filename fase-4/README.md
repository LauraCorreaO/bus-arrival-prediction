# Fase 4 · Monitoreo

- Código de registro: `fase-3/src/registro.py` y el middleware `registrar_predicciones` de `fase-3/api.py`.
  Cada intento de `/predict` agrega una fila `fecha_hora,estado_http,version,prediccion`, incluidos los rechazos 400/422
  (en un error la predicción queda vacía y, si no se cargó modelo, también la versión).
- Script: `fase-4/monitoreo.py` (total de solicitudes y porcentaje de fallos 4xx/5xx).
- Muestra del registro: `fase-4/muestra_registro.csv`.

## Inicio con el registro habilitado (desde `fase-3/`)

```powershell
docker run --rm -p 8000:8000 -v "${PWD}/models:/app/models" -v "${PWD}/registro:/app/registro" -e MODELO_VERSION=v1 bus-api:fase3
```

El registro persistente queda en `fase-3/registro/solicitudes.csv` del anfitrión (`/app/registro/solicitudes.csv` en el contenedor).
Sin Docker se escribe en la misma ruta, o en la que indique `REGISTRO_PATH`.

```powershell
python fase-4/monitoreo.py fase-3/registro/solicitudes.csv
```

## Reporte

Ejecución del 22-11-2026 sobre un registro vacío, con el contenedor y `MODELO_VERSION=v1` (`python cliente.py --sin-campo empresa`):

| # | Solicitud | Respuesta |
|---|---|---|
| 1 | Viaje Jericó → Terminal del Sur, automóvil, 14:00, domingo, julio | 200, 2,49 h, v1 |
| 2 | Viaje Guarne → Terminal del Norte, bus, 11:40, lunes, marzo | 200, 0,96 h, v1 |
| 3 | El viaje 1 sin el campo `empresa` | 422, «Falta el campo obligatorio 'empresa'» |

Salida del script:

```text
Total de solicitudes: 3
Solicitudes fallidas (4xx/5xx): 1
Porcentaje de fallos: 33,3 %
```

La API procesó dos registros con v1 y rechazó el tercero porque faltaba `empresa`; la tasa de fallos fue 1/3 (33,3 %),
el resultado esperado de la prueba de validación. Este indicador describe el funcionamiento del servicio, no la calidad predictiva.

## Versiones

- `v1` (`fase-3/models/v1/`): versión activa de la API (`MODELO_VERSION=v1`, valor por defecto).
- `v2` (`fase-3/models/v2/`): versión anterior disponible de la fase 2.
- `v3` (`fase-3/models/v3/`): creada con `POST /train` sin cuerpo (datos completos, semilla 42). Reproduce las métricas
  de v1 (MAE 0,4322 h); no se activa porque no mejora a la versión activa, según la política de abajo.

Para volver a la otra versión basta reiniciar la API con `-e MODELO_VERSION=v2` (o `v1`); no se borra ni modifica ninguna carpeta de `models/`.

## Política de reentrenamiento

Cada trimestre, o al reunir 50 000 viajes nuevos con hora de llegada registrada en datos.gov.co, el equipo ejecutará `POST /train`
con un identificador nuevo. Comparará la candidata y la versión activa con la MAE (métrica principal de la fase 2) sobre los
mismos viajes reservados más recientes, y cambiará `MODELO_VERSION` a la candidata solo si reduce la MAE; si no, se mantiene la activa.
