## Gestión de versiones del modelo

Cada versión entrenada (mediante `train.py` o el endpoint `POST /train`) se almacena en su propia carpeta dentro de `fase-3/models/<version>/` con tres artefactos inmutables (`modelo.joblib`, `metrics.json` y `metadata.json`). Si el identificador ya existe, la operación se rechaza sin sobrescribir artefactos previos.

### Cómo ver las versiones disponibles

1. **Listar las carpetas de versiones creadas** (desde `fase-3/`):
   ```powershell
   dir models