# Places retrieval v1

Dataset JSONL sintético para arrancar el bi-encoder E5 de Places. Cada fila contiene:

- `query`: consulta sin prefijo E5.
- `positive`: documento con formato `structured-place-v3`.
- `hard_negatives`: tres lugares plausibles pero incorrectos del mismo split.

Los splits no comparten queries ni documentos positivos. No agregues `query:` o
`passage:` a los archivos: el script de entrenamiento aplica esos prefijos.

Conteos:

- `train.jsonl`: 160 pares.
- `validation.jsonl`: 20 pares de conceptos no vistos.
- `test.jsonl`: 20 pares de conceptos no vistos.
- Total: 200 registros.

Las sedes derivadas tienen nombres, documentos, consultas y atributos discriminantes
propios. Variantes del mismo concepto se excluyen de los hard negatives explícitos
para no etiquetar como incorrecto un lugar que también podría satisfacer la consulta.
Validation y test conservan un solo documento por cada concepto no visto; así sus positivos
no compiten contra una sede hermana igualmente relevante durante la evaluación.

Este corpus cubre vocabulario abierto y sirve para iniciar el fine-tuning, pero no es
suficiente por sí solo para una decisión productiva. Sustituir o ampliar train con
lugares reales y pares de relevancia anonimizados/adjudicados. Mantener validation y
test fuera del entrenamiento y medir Recall@k, MRR y nDCG@k.

Regeneración reproducible:

```bash
python scripts/build_place_training_datasets.py
```
