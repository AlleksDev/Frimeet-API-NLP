# Places retrieval v1

Dataset JSONL sintético para arrancar el bi-encoder E5 de Places. Cada fila contiene:

- `query`: consulta sin prefijo E5.
- `positive`: documento con formato `structured-place-v3`.
- `hard_negatives`: lugares plausibles pero incorrectos del mismo split. Train
  contiene tres; validation y test contienen siete para evitar evaluaciones triviales.
- `challenge_tags` (validation y test): tipo de dificultad de la consulta, por ejemplo
  `colloquial`, `misspelling`, `implicit`, `contrastive` o `constraint`.

Los splits no comparten queries ni documentos positivos. No agregues `query:` o
`passage:` a los archivos: el script de entrenamiento aplica esos prefijos.

Conteos:

- `train.jsonl`: 160 pares.
- `validation.jsonl`: 60 consultas sobre 20 conceptos no vistos (tres formulaciones
  por documento relevante).
- `test.jsonl`: 80 consultas sobre 20 conceptos no vistos (cuatro formulaciones
  por documento relevante).
- Total: 300 registros; el split usado para entrenar permanece en 160 pares.

Las sedes derivadas tienen nombres, documentos, consultas y atributos discriminantes
propios. Variantes del mismo concepto se excluyen de los hard negatives explícitos
para no etiquetar como incorrecto un lugar que también podría satisfacer la consulta.
Validation y test conservan un solo documento por cada concepto no visto; así sus positivos
no compiten contra una sede hermana igualmente relevante durante la evaluación. Test repite
el documento relevante para consultas canónicas, coloquiales, implícitas y con errores de
escritura. Eso es intencional: varios intents lingüísticos pueden compartir el mismo qrel.
Validation también contiene reformulaciones difíciles para elegir épocas y checkpoints sin
consultar repetidamente el holdout final.

No evalúes cada fila únicamente contra sus `hard_negatives`. Usa el corpus global
deduplicado para que las 80 consultas compitan contra los 20 documentos de test:

```bash
python scripts/evaluate_place_retriever.py \
  --test-file data/training/places_retrieval_v1/test.jsonl \
  --model base=intfloat/multilingual-e5-base \
  --model fine_tuned=/ruta/al/modelo
```

El resultado incluye Top-1, Recall@3/5/10, MRR, nDCG@10 y métricas separadas por
`challenge_tags`. Elige épocas con validation; usa test solo para la comparación final.

Este corpus cubre vocabulario abierto y sirve para iniciar el fine-tuning, pero no es
suficiente por sí solo para una decisión productiva. Sustituir o ampliar train con
lugares reales y pares de relevancia anonimizados/adjudicados. Mantener validation y
test fuera del entrenamiento y medir Recall@k, MRR y nDCG@k.

Regeneración reproducible:

```bash
python scripts/build_place_training_datasets.py
```
