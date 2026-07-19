# Places intent v1

Dataset JSONL sintético para arrancar el token classifier de `/places/chat`.
Cada registro contiene `text` y spans `[start, end)` con uno de estos slots:
`CATEGORY`, `PREFERENCE`, `EXCLUSION`, `LOCATION`, `REFERENCE` o `RADIUS`.

Los valores de `CATEGORY` son texto abierto y varias categorías de validación/test
no aparecen en train. El corpus no contiene conversaciones ni datos personales reales.

Archivos:

- `train.jsonl`: 132 ejemplos de entrenamiento.
- `validation.jsonl`: 34 ejemplos para selección de checkpoint por pérdida.
- `test.jsonl`: 34 ejemplos de evaluación final; no usar durante ajuste.

Total: 200 registros. Incluye errores ortográficos controlados y categorías no
vistas fuera de train, sin convertir esas frases en etiquetas de clase.
El slot `REFERENCE` tiene ejemplos combinados con categorías abiertas para evitar
que quede subrepresentado frente a ubicación, preferencia y radio.

Antes de producción, complementar con mensajes reales anonimizados y doblemente
revisados. Medir precision, recall y F1 por slot; no aprobar usando sólo train loss.

Regeneración reproducible:

```bash
python scripts/build_place_training_datasets.py
```
