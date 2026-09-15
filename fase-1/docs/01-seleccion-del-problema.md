# Selección del problema — Fase 1

**Dataset:** Llegadas y Salidas de Vehículos de la Terminal de Transporte
**Fuente:** Portal de Datos Abiertos de Colombia — recurso `pfsr-mdyi`
**Endpoint usado:** `https://www.datos.gov.co/resource/pfsr-mdyi.json` (consulta vía SoQL, sin necesidad de CSV)
**Fecha de verificación:** 2026-09-15
**Alcance verificado:** filtro `clase_vehiculo = BUS`, consultado directamente contra el dataset completo publicado en el portal (no una muestra local)

---

## 0. ¿Es necesario el CSV?

**No.** El endpoint responde a consultas SoQL (`$query`, `$limit`, `$offset`, agregaciones `count`/`sum`/`GROUP BY`, etc.). Se puede traer todo lo necesario directamente a un DataFrame de pandas desde el notebook de la Fase 1, sin depender de la descarga manual desde el portal. Esto ya se probó y funciona (se consultaron agregados sobre el dataset completo, ~1.3M de filas).

Nota técnica: por defecto la API solo devuelve 1.000 filas si no se especifica `$limit`. Para traer más hay que paginar con `$limit`/`$offset`, o usar agregaciones (`count`, `GROUP BY`, `sum`) que sí se calculan sobre la tabla completa sin necesidad de descargar fila por fila.

---

## 1. Hallazgo importante: este recurso solo contiene LLEGADAS

Se revisaron los valores distintos de la columna `estado` sobre las 1.323.373 filas de clase `BUS`:

| estado | filas |
|---|---|
| LLEGADA | 923.310 |
| Llegada | 400.063 |

Son el mismo valor con inconsistencia de mayúsculas/minúsculas — es decir, **`estado` es constante en la práctica**. Este recurso del portal (`pfsr-mdyi`) contiene únicamente registros de llegada, no de salida, a pesar del nombre del dataset. Por lo tanto:

- `estado` **no sirve** como variable objetivo ni como predictora (varianza cero).
- Se descarta en la limpieza (a confirmar formalmente en la exploración, punto 8).

---

## 2. Variable objetivo — CONFIRMADA con el profesor

**`duracion_viaje_horas`** — tiempo transcurrido entre la salida y la llegada del viaje, en horas (decimal).

```
duracion_viaje_horas = (fecha_llegada - fecha_salida).total_seconds() / 3600
```

**Tipo de problema: Regresión.** Es el problema clásico de tiempo estimado de llegada (ETA), muy usado en modelos de movilidad (el mismo tipo de problema que resuelven Google Maps o Uber).

### Justificación frente a la objeción "los datos ya nos dicen cuánto se va a demorar, porque tenemos la hora de salida y de llegada"

- Es cierto que en los datos **históricos** ya conocemos ambas marcas de tiempo — pero eso pasa en *cualquier* problema supervisado: la respuesta correcta siempre está en los datos de entrenamiento, por eso se puede entrenar el modelo.
- La pregunta relevante no es "¿ya sabemos la duración de viajes pasados?" (sí, siempre), sino "¿sabemos la duración de un viaje que **apenas está saliendo**, antes de que llegue?" — ahí la respuesta es no. Ese es el problema real que resolvería el modelo en producción: estimar cuánto tardará un viaje usando solo lo que se conoce al momento de la salida (ruta, empresa, hora, día, terminal), antes de que exista la hora de llegada.

**Consultado con el profesor: aprobado**, con la condición de hacer primero la exploración de datos antes de decidir qué columnas se usan o se descartan (ver punto 8).

### Condición obligatoria para que sea válido (evitar fuga de información)

- `fecha_llegada` y `hora_de_llegada` **no pueden usarse como predictoras** bajo ningún motivo — solo sirven para calcular el objetivo. Si se incluyeran, el modelo "vería" la respuesta disfrazada (*data leakage*).
- `pasajeros` sí puede usarse como predictora en este escenario (no hay fuga: el número de pasajeros no depende de cuánto se demoró el viaje).

*(Variable descartada como objetivo: `pasajeros` — sigue siendo válida y más "segura" por no admitir la objeción anterior, pero el profesor aprobó duración, así que se usa como predictora en su lugar.)*

*(Variable descartada como objetivo: `estado` — no es viable porque es constante, ver punto 1.)*

---

## 3. Variables predictoras candidatas (mínimo 5, sin contar el objetivo)

Esta es la lista **preliminar**, antes de la exploración de datos. La lista final se confirma en el punto 8, después de ver estadísticas, nulos e inconsistencias — tal como indicó el profesor (primero se explora, luego se decide qué se descarta).

| # | Variable | Tipo | Origen |
|---|---|---|---|
| 1 | `empresa` | Categórica nominal | Original |
| 2 | `nombre_sucursal` | Categórica nominal | Original |
| 3 | `ruta_origen` | Categórica nominal | Original |
| 4 | `ruta_destino` | Categórica nominal | Original |
| 5 | `subregi_n` | Categórica nominal | Original |
| 6 | `pasajeros` | Numérica **entera** | Original |
| 7 | `dia_semana` (0–6) | Numérica **entera** | Derivada de `fecha_salida` |
| 8 | `hora_salida_decimal` (ej. 14.5) | Numérica **decimal (float)** | Derivada de `fecha_salida` (hora + minuto/60) |
| 9 | `fecha_hora_salida_origen` | Fecha/hora (a decidir cómo se usa) | Original — tiene 33% de nulos, ver punto 5 |

Son 9 candidatas (5 categóricas + 2 enteras + 1 decimal + 1 a definir), muy por encima del mínimo de 5. Ya se cumple la mezcla de enteros, decimales y categóricas exigida por el profesor sin depender de `fecha_hora_salida_origen`.

---

## 4. La variable decimal

Con la duración como objetivo, el decimal ya no puede salir de restar fechas de llegada (eso sería el target). Se usa en su lugar:

```
hora_salida_decimal = hora + minuto/60   # ej. 14:30 -> 14.5
```

Calculada a partir de `fecha_salida`, que tiene prácticamente cero nulos (5 de 1.323.373 = 0.0004%). Es un decimal real (no un entero disfrazado) y no depende de `fecha_llegada`, así que no genera fuga de información.

---

## 5. Valores nulos — el profesor confirmó que el 33% está bien

Se midió el conteo real de nulos sobre el dataset completo de BUS (1.323.373 filas), no sobre una muestra:

| Columna | Nulos | % |
|---|---|---|
| `fecha_hora_salida_origen` | 444.325 | **33.57%** (en 2024 sube a 97.3%, concentrado ahí) |
| `empresa` | 19 | 0.0014% |
| `fecha_salida` | 5 | 0.0004% |
| `nombre_sucursal`, `fecha_llegada`, `hora_de_llegada`, `ruta_origen`, `ruta_destino`, `subregi_n`, `pasajeros`, `estado` | 0 | 0% |

**Actualización tras hablar con el profesor:** no es necesario que el porcentaje caiga en un rango específico (0.1%–2%). Confirmó que **un 33% de nulos en `fecha_hora_salida_origen` está bien, siempre que se le dé un manejo adecuado y documentado** (por ejemplo: imputar, crear una variable indicadora de "faltante", o justificar por qué se descarta la columna). Ya **no hace falta simular nulos artificiales** — se descarta esa idea.

Qué hacer con `fecha_hora_salida_origen` se decide en la exploración (punto 8): puede terminar siendo predictora (con imputación), predictora binaria de "faltante", o descartada si resulta redundante con `fecha_salida`.

---

## 6. Tamaño del dataset

- Total de filas BUS en el dataset completo: **1.323.373** (muy por encima del mínimo de 1.000).
- Para que sea manejable en un computador personal (requisito del enunciado), se recomienda trabajar con una **muestra** en la Fase 1 en lugar del dataset completo — por ejemplo, 150.000–200.000 filas tomadas de 2023–2025 (los años con volumen real; unas ~30 filas sueltas de años como 2002–2012 y 2029 son errores de captura y deben descartarse en la limpieza).

---

## 7. ¿Es un problema de series de tiempo? (No debe serlo)

**No lo es, siempre que se modele como se propone aquí.**

- Cada fila es un viaje individual e independiente (un bus, una ruta, una fecha/hora puntual).
- El objetivo (`duracion_viaje_horas`) se predice a partir de atributos propios de ESE viaje (ruta, empresa, hora, día de la semana, pasajeros), no a partir de valores pasados de la serie ni con el fin de pronosticar un valor futuro en una secuencia temporal continua.
- Es un problema de **regresión tabular / transversal (cross-sectional)**, no de forecasting.

Para que se mantenga así durante la Fase 1, hay que evitar:
- Usar variables de *lag* (duración del viaje anterior, promedio móvil, etc.).
- Ordenar las filas por tiempo y dividir train/test de forma secuencial pensando en "predecir el futuro".
- Agregar los datos por intervalos de tiempo (ej. duración promedio por día) para pronosticar el siguiente periodo — eso sí sería series de tiempo, y no es lo que se va a hacer.

La fecha/hora se usa únicamente para **derivar** features puntuales del viaje (hora del día, día de la semana), tratando cada viaje como una observación independiente. Con eso, el split train/test puede ser aleatorio (no necesita ser temporal), típico de un problema de regresión estándar.

---

## 8. Próximo paso: Exploración de Datos (EDA) — antes de decidir columnas

Instrucción del profesor (parafraseada): **primero se hace la exploración de los datos, y luego —con base en esa exploración— se decide qué variables se descartan o cómo se tratan.** No se descarta nada por adelantado sin haberlo explorado.

Se usará como referencia la metodología de [ai4eng — notes 04-01 Data Exploration](https://rramosp.github.io/ai4eng.v1/content/notes-04-01-data-exploration/), que propone esta secuencia:

1. Cargar datos e inspeccionar dimensiones.
2. Contar valores faltantes por columna.
3. Analizar la variable objetivo (`duracion_viaje_horas`): distribución, outliers, valores negativos o absurdos.
4. Identificar tipos de datos (numéricas vs. categóricas).
5. Generar estadísticas descriptivas de las numéricas (media, mínimo, máximo, desviación estándar, cuartiles).
6. Calcular frecuencias de las categóricas (conteo por categoría, cardinalidad).
7. Matriz de correlación entre numéricas.
8. Visualizar patrones de valores nulos (ej. con `missingno`), especialmente en `fecha_hora_salida_origen`.

**Salida de este paso:** un notebook de exploración en `fase-1/notebooks/` (sin modelar todavía), que documente los hallazgos y sirva de base para las decisiones de limpieza (qué columnas se quedan, cuáles se descartan por redundancia, cómo se tratan los nulos de `fecha_hora_salida_origen`, cómo se filtran los ~30 registros de años atípicos, etc.).

Este es el siguiente paso a ejecutar.

---

## Resumen de cumplimiento

| Requisito | Estado |
|---|---|
| Variable objetivo clara | ✅ `duracion_viaje_horas` (confirmada con el profesor) |
| Clasificación o regresión | ✅ Regresión |
| ≥1.000 observaciones | ✅ 1.323.373 (se recomienda muestrear ~150k–200k) |
| ≥5 predictoras | ✅ 9 candidatas (lista final tras EDA) |
| Variables numéricas (entero y decimal) y categóricas | ✅ Enteras (`pasajeros`, `dia_semana`), decimal (`hora_salida_decimal`), categóricas (5) |
| No es series de tiempo | ✅ Si se modela como regresión tabular por viaje (ver punto 7) |
| Predictora con datos faltantes | ✅ `fecha_hora_salida_origen` (33.57%), aprobado por el profesor con manejo documentado |
| Procesable en computador personal | ✅ Con muestreo (punto 6) |

---

## Decisiones confirmadas con el profesor

1. Variable objetivo: **`duracion_viaje_horas`** (regresión / ETA).
2. El 33% de nulos en `fecha_hora_salida_origen` **no es un problema**, siempre que se documente su manejo.
3. **Antes de eliminar o transformar ninguna columna, se hace la exploración de datos (EDA).** Las decisiones de limpieza salen de esa exploración, no al revés.

## Siguiente paso

Construir el notebook de exploración de datos en `fase-1/notebooks/` siguiendo la secuencia del punto 8.
