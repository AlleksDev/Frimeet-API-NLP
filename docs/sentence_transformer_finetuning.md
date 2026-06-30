# Fine-tuning Del Encoder De Recuperacion

## Decision Tecnica

El encoder base de produccion es `intfloat/multilingual-e5-small`:

- fue preentrenado para recuperacion semantica, no solo para similitud de frases;
- incluye espanol dentro de su entrenamiento multilingue;
- genera vectores de 384 dimensiones;
- su tamano es razonable para inferencia CPU en el Space;
- usa entradas asimetricas: `query: ...` para consultas y `passage: ...` para documentos.

El modelo `hiiamsid/sentence_similarity_spanish_es` de Lab6 sigue siendo valido como
demostracion academica de Sentence-BERT, pero produce 768 dimensiones y su objetivo
base es similitud de oraciones. Para este servicio conviene partir de un modelo ya
orientado a retrieval y afinarlo con los juicios de relevancia de Frimeet.

Referencias primarias:

- [Repositorio oficial E5 de Microsoft](https://github.com/microsoft/unilm/tree/master/e5)
- [Reporte tecnico Multilingual E5](https://arxiv.org/abs/2402.05672)
- [MultipleNegativesRankingLoss](https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss)

## Alcance Y Limite Espacial

Este cambio no agrega NER ni interpreta entidades geograficas escritas en la consulta.
El fine-tuning si puede aprender que en `cafeteria cerca del Parque Central` la clase
objetivo es una cafeteria y que el parque es un distractor tematico. No puede comprobar
por si mismo que un establecimiento este fisicamente cerca de ese parque.

La cercania real continua resolviendose con `lat`, `lng` y `radius`: la API principal
calcula los IDs cercanos y PGVector se limita a ese conjunto. No se inventan
coordenadas ni se guardan en el indice semantico.

## Dataset De Entrenamiento

### Formato

Usar UTF-8 JSONL: un objeto por linea y un triplete por hard negative.

| Campo | Regla |
|---|---|
| `query_id` | ID estable y unico de la intencion. La misma consulta no puede cruzar splits. |
| `query` | Texto natural escrito como lo haria un usuario, sin prefijo `query:`. |
| `positive_id` | ID real del lugar relevante. |
| `positive` | Documento semantico exacto que se indexara para ese lugar. |
| `negative_id` | ID real de un candidato no relevante pero dificil. |
| `negative` | Documento semantico exacto del hard negative. |
| `negative_type` | Taxonomia del error que se quiere corregir. |
| `split` | `train`, `validation` o `test`. |

Ejemplo:

```json
{"query_id":"q-cafe-001","query":"quiero una cafeteria cerca del parque central","positive_id":"218","positive":"cafeteria bebidas desayuno postres ...","negative_id":"441","negative":"parque naturaleza caminar ...","negative_type":"spatial_reference_wrong_type","split":"train"}
```

El archivo `examples/training/place_retrieval.example.jsonl` solo ilustra el formato.
Sus IDs son ficticios y no debe usarse para entrenar el modelo desplegado.

Para crear una primera propuesta balanceada desde el corpus real:

```powershell
python scripts/build_place_retrieval_labels.py `
  --input ..\place_corpus.jsonl `
  --output ..\frimeet_retrieval_weak_labels.jsonl `
  --review-csv ..\frimeet_retrieval_review.csv `
  --report ..\frimeet_retrieval_labeling_report.json
```

El JSONL generado ya cumple el esquema, pero contiene etiquetas debiles. Abrir el CSV,
revisar positivo y negativo, y cambiar `review_status` de `pending` a `approved` o
`rejected`. Después construir el dataset final:

```powershell
python scripts/finalize_place_retrieval_labels.py `
  --labels ..\frimeet_retrieval_weak_labels.jsonl `
  --review-csv ..\frimeet_retrieval_review.csv `
  --output ..\frimeet_retrieval_final.jsonl
```

El finalizador se detiene si queda alguna fila pendiente y vuelve a validar splits,
documentos e IDs antes de permitir el entrenamiento.

### Como Obtener Los Textos

Los textos positivo y negativo deben salir de la misma funcion que alimenta PGVector:
`place_to_source_record(...).document`. No conviene redactar resúmenes manuales porque
el modelo aprenderia contra una representacion distinta de la que vera en produccion.

Para construir los juicios:

1. Exportar lugares reales desde la API principal con
   `python scripts/export_place_retrieval_corpus.py --output /content/place_corpus.jsonl`.
2. Escribir consultas reales o anonimizadas del producto.
3. Recuperar 10-30 candidatos con el modelo base.
4. Etiquetar relevancia `3` (ideal), `2` (relevante), `1` (posible) o `0` (no relevante).
5. Crear positivos solo con grados `2` o `3`.
6. Elegir hard negatives con grado `0`, preferentemente entre los candidatos mejor
   posicionados por el modelo base.
7. Reservar grado `1` para evaluacion; no usarlo como negativo fuerte porque es ambiguo.

### Tipos De Hard Negative Requeridos

Cubrir al menos estas clases:

- `spatial_reference_wrong_type`: el nombre de la referencia coincide, pero el tipo
  de lugar no satisface la necesidad;
- `spatial_context_distractor`: lugar relacionado con la zona, no con la intencion;
- `lexical_overlap_wrong_intent`: comparte palabras como `cafe`, pero es un museo y
  no una cafeteria;
- `popular_category_bias`: restaurante popular para una consulta que no pide comida;
- `same_category_wrong_tags`: misma categoria general, experiencia incorrecta;
- `same_intent_wrong_constraint`: intencion correcta, pero falla ambiente, precio o
  actividad solicitada;
- `ambiguous_name`: el nombre parece coincidir, pero categoria, tags y descripcion no;
- `near_duplicate`: dos lugares muy parecidos donde solo uno cumple la necesidad.

Para frases espaciales, el positivo debe cumplir el **tipo de lugar solicitado** y el
hard negative puede ser la referencia espacial. Sin coordenadas verificadas no se debe
etiquetar `cerca` o `lejos`; eso le corresponde al filtro geografico.

### Cobertura Y Tamano

Recomendacion practica, no garantia estadistica:

- prueba tecnica: 100 consultas unicas y unas 300 tripletas;
- primera version util: 500-1,000 consultas unicas, con 2-5 hard negatives por consulta;
- version robusta: 3,000 o mas consultas, revisadas y balanceadas por categoria/tag.

Mantener ejemplos de cafeteria, restaurante, arte/cultura, parques, deporte, compras,
comunidad, ocio nocturno, hospedaje y lugares familiares. Incluir faltas ortograficas,
sinonimos regionales, consultas cortas, nombres ambiguos y negaciones. Evitar que la
cantidad de restaurantes domine el dataset.

Dividir por `query_id`, no por filas: 80% train, 10% validation y 10% test es un buen
punto de partida. El mismo texto o parafrasis cercana no debe aparecer en dos splits.
El validador tambien exige que un ID de lugar conserve siempre el mismo documento.

## Validar Y Entrenar En Google Colab

Activar una GPU T4 y clonar la rama:

```python
!git clone --branch hf-deploy https://github.com/AlleksDev/Frimeet-API-NLP.git
%cd Frimeet-API-NLP
!pip install -q -r requirements-training.txt
```

Subir el JSONL real, por ejemplo a `/content/frimeet_retrieval.jsonl`, y validarlo:

```python
!python scripts/validate_retrieval_dataset.py /content/frimeet_retrieval.jsonl
```

Configurar `HF_TOKEN` en Secrets de Colab con permiso de escritura y entrenar:

```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

```python
!python scripts/train_retrieval_encoder.py \
  --dataset /content/frimeet_retrieval.jsonl \
  --output-dir /content/frimeet-e5-places-v1 \
  --epochs 3 \
  --batch-size 16 \
  --hub-model-id AlleksDev/frimeet-e5-places-v1
```

El script mide la linea base y el modelo afinado con retrieval metrics sobre
`validation`, guarda un SHA-256 del dataset, los hiperparametros y ambas mediciones en
`frimeet_embedding_config.json`, y finalmente sube el artefacto si se especifica
`--hub-model-id`. No desplegar si nDCG/MRR empeoran de forma consistente.

## Migrar PGVector: VECTOR(300) A VECTOR(384)

Se debe hacer `TRUNCATE` y recargar, no `UPDATE`: FastText y E5 no comparten espacio
vectorial y ademas cambia la dimension.

Orden seguro:

1. Entrenar y subir primero el modelo.
2. Crear snapshot de RDS.
3. Pausar Space y jobs.
4. En pgAdmin ejecutar completo
   `sql/pgadmin_migrate_sentence_transformer_384.sql`.
5. Recargar lugares y posts con los scripts Colab.
6. Ejecutar `sql/verify_sentence_transformer_embeddings.sql`.
7. Actualizar variables del Space y desplegar.

La migracion aborta si no encuentra ambas columnas en `VECTOR(300)`, trunca solamente
las dos tablas derivadas, recrea HNSW y las funciones con `VECTOR(384)`, y reaplica
permisos. Al verificar se esperan dimensiones 384 y normas cercanas a 1.

Para la recarga, usar un runtime Colab con GPU T4. En **Secrets** configurar y
habilitar acceso para:

- `PGVECTOR_WRITER_PASSWORD` (obligatorio);
- `EMBEDDING_MODEL=AlleksDev/frimeet-e5-places-v1`;
- `EMBEDDING_VERSION=frimeet-e5-places-v1`;
- `HF_TOKEN` si el modelo es privado;
- `MAIN_API_INTERNAL_TOKEN` si la API principal lo exige.

Luego ejecutar:

```python
!git clone --branch hf-deploy https://github.com/AlleksDev/Frimeet-API-NLP.git
%cd Frimeet-API-NLP
!python scripts/colab_initial_load_places.py
!python scripts/colab_initial_load_posts.py --skip-install --skip-download
```

Los dos jobs y el Space deben usar exactamente el mismo `EMBEDDING_MODEL`,
`EMBEDDING_VERSION`, dimension y prefijos. Si todavia no existe un dataset real
aprobado, se puede cargar primero el modelo base dejando fuera los Secrets
`EMBEDDING_MODEL`/`EMBEDDING_VERSION`; no se debe presentar esa variante como modelo
afinado.

## Variables Del Space

```env
EMBEDDING_PROVIDER=sentence_transformer
EMBEDDING_DIMENSION=384
EMBEDDING_MODEL=AlleksDev/frimeet-e5-places-v1
EMBEDDING_VERSION=frimeet-e5-places-v1
SENTENCE_TRANSFORMER_REVISION=main
SENTENCE_TRANSFORMER_AUTO_DOWNLOAD=true
SENTENCE_TRANSFORMER_QUERY_PREFIX="query: "
SENTENCE_TRANSFORMER_DOCUMENT_PREFIX="passage: "
SENTENCE_TRANSFORMER_BATCH_SIZE=32
SENTENCE_TRANSFORMER_DEVICE=cpu
SENTENCE_TRANSFORMER_MAX_SEQUENCE_LENGTH=256
SEMANTIC_NO_MATCH_THRESHOLD=0.70
SEMANTIC_RELEVANCE_THRESHOLD=0.80
```

Si el repositorio del modelo es privado, agregar `HF_TOKEN` como **Secret** de lectura.
Si es publico, no se necesita un secreto nuevo. Las credenciales de PGVector, API
principal y Groq existentes no cambian.

Los umbrales 0.70/0.80 son valores iniciales conservadores para E5, cuyas similitudes
absolutas suelen estar mas concentradas que las de FastText. Deben calibrarse con el
split test y consultas sin respuesta relevante; no comparar sus valores numericos con
los scores del modelo anterior.

El Dockerfile precarga el E5 base para que el Space siempre pueda iniciar. Si
`EMBEDDING_MODEL` apunta al modelo afinado, este se descarga en el primer arranque. Para
evitar esa descarga en cada reconstruccion, cambiar el valor por defecto del ARG
`SENTENCE_TRANSFORMER_MODEL_REPO_ID` en el Dockerfile al repositorio afinado.

## Compatibilidad HTTP

No cambia ningun body, campo, ruta ni status code de los endpoints. Solamente cambia
como se generan los vectores de consultas/documentos y, como consecuencia esperada,
el orden y los scores de similitud.
