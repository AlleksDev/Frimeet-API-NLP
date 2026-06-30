---
title: Frimeet API NLP
sdk: docker
app_port: 7860
pinned: false
---

# Frimeet API NLP

Servicio NLP independiente para busqueda semantica con Sentence Transformers/E5 + pgvector, recomendaciones y redaccion conversacional con Llama via Groq.

La API principal sigue siendo la fuente de verdad de lugares, posts, usuarios, sesiones, permisos y reportes. Este servicio NLP solo trabaja con datos derivados para busqueda semantica.

## Arquitectura

```text
API principal
  |-- fuente de verdad de places/posts
  `-- consume la API NLP por REST

Hugging Face API NLP
  |-- usa credenciales nlp_reader
  |-- consulta RDS PostgreSQL + pgvector
  |-- genera el embedding E5 del query del usuario
  |-- ordena por similitud coseno en pgvector
  `-- usa Groq/Llama para embellecer recomendaciones y chat

Hugging Face Jobs
  |-- usan credenciales nlp_writer
  |-- consumen endpoints paginados/cursor de la API principal
  |-- generan embeddings por batch
  `-- hacen upsert via funciones SQL controladas

RDS PostgreSQL + pgvector
  |-- place_embeddings
  |-- post_embeddings
  |-- match_places
  |-- match_posts
  |-- get_place_content_hashes
  |-- get_post_content_hashes
  |-- upsert_place_embedding
  `-- upsert_post_embedding
```

## Variables De Entorno

```env
ENV=local
API_HOST=0.0.0.0
API_PORT=8080

MAIN_API_BASE_URL=http://3.212.166.108
MAIN_API_PLACES_SEARCH_PATH=/api/v1/places/search
MAIN_API_PLACES_NEARBY_PATH=/api/v1/places/nearby
MAIN_API_POSTS_SEARCH_PATH=/api/v1/posts/search
MAIN_API_INTERNAL_TOKEN=
MAIN_API_TIMEOUT_SECONDS=15
MAIN_API_PLACES_PAGE_LIMIT=100
MAIN_API_POSTS_PAGE_LIMIT=100
MAIN_API_PLACES_PAGINATION_MODE=cursor
MAIN_API_POSTS_PAGINATION_MODE=cursor

GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant

VECTOR_STORE_PROVIDER=aws_pgvector

PGVECTOR_HOST=nlp-vector-db.c2jwncm87zsa.us-east-1.rds.amazonaws.com
PGVECTOR_PORT=5432
PGVECTOR_DATABASE=nlp_vectors
PGVECTOR_READER_USER=nlp_reader
PGVECTOR_READER_PASSWORD=CAMBIA_ESTA_PASSWORD_READER
PGVECTOR_WRITER_USER=nlp_writer
PGVECTOR_WRITER_PASSWORD=CAMBIA_ESTA_PASSWORD_WRITER
PGVECTOR_SSL_MODE=require

EMBEDDING_PROVIDER=sentence_transformer
EMBEDDING_DIMENSION=384
EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_VERSION=multilingual-e5-small-base-v1
SENTENCE_TRANSFORMER_MODEL_PATH=
SENTENCE_TRANSFORMER_CACHE_DIR=.models/sentence-transformers
SENTENCE_TRANSFORMER_REVISION=main
SENTENCE_TRANSFORMER_AUTO_DOWNLOAD=true
SENTENCE_TRANSFORMER_QUERY_PREFIX="query: "
SENTENCE_TRANSFORMER_DOCUMENT_PREFIX="passage: "
SENTENCE_TRANSFORMER_BATCH_SIZE=32
SENTENCE_TRANSFORMER_DEVICE=cpu
SENTENCE_TRANSFORMER_MAX_SEQUENCE_LENGTH=256
FASTTEXT_MODEL_PATH=.models/fasttext-es/model.bin
FASTTEXT_MODEL_REPO_ID=facebook/fasttext-es-vectors
FASTTEXT_MODEL_FILENAME=model.bin
FASTTEXT_AUTO_DOWNLOAD=true

BM25_K1=1.5
BM25_B=0.75
BM25_RELEVANCE_THRESHOLD=3.0
SEMANTIC_NO_MATCH_THRESHOLD=0.70
SEMANTIC_RELEVANCE_THRESHOLD=0.80

LOG_LEVEL=INFO
REQUEST_TIMEOUT_SECONDS=10
LLM_TIMEOUT_SECONDS=12
MAX_LLM_CONCURRENT_REQUESTS=4
EMBEDDING_CACHE_TTL_SECONDS=300
VECTOR_SEARCH_CACHE_TTL_SECONDS=120
```

La API crea el cliente RDS con rol `reader`. Los jobs crean el cliente con rol `writer`. Si configuras `PGVECTOR_READER_*` y `PGVECTOR_WRITER_*` en el mismo entorno, el codigo elige automaticamente las credenciales correctas para cada flujo.

En Hugging Face, guarda passwords y tokens como Secrets.

## Cache Local Efimera

La API usa cache local en memoria para:

- embeddings de queries repetidas,
- resultados de busqueda/recomendacion frecuentes.

Este cache vive solo mientras el contenedor de Hugging Face este despierto. Si el Space se duerme o reinicia, se pierde sin problema porque RDS sigue siendo la fuente persistente de embeddings derivados.

## Instalacion Local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

## Ejecutar API

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
```

Documentacion local:

```text
http://127.0.0.1:8080/docs
```

## Hugging Face Docker Space

El proyecto incluye `Dockerfile` para Hugging Face Spaces. El contenedor expone el puerto `7860` y arranca:

```bash
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}
```

Durante el build, Docker precarga `intfloat/multilingual-e5-small` en una capa
cacheada. El modelo afinado se puede seleccionar con `EMBEDDING_MODEL`; consulta
`docs/sentence_transformer_finetuning.md` para entrenamiento y despliegue.

## Endpoints

```http
GET  /health
GET  /ready
POST /places/search
GET  /places/search/metrics?k=5
POST /places/search/metrics?k=5
POST /places/recommendations
POST /places/chat
POST /posts/recommendations
GET  /posts/clusters
```

## Sync Jobs

Probar sin escribir:

```powershell
python -m app.jobs.sync_place_embeddings --dry-run --max-pages 1
python -m app.jobs.sync_post_embeddings --dry-run --max-pages 1
```

Primera carga:

```powershell
python -m app.jobs.initial_load_place_embeddings
python -m app.jobs.initial_load_post_embeddings
```

Sincronizaciones posteriores:

```powershell
python -m app.jobs.sync_place_embeddings
python -m app.jobs.sync_post_embeddings
```

Los jobs calculan un `content_hash` versionado con el contenido, modelo, version y
dimension. Un cambio de modelo fuerza la regeneracion aunque el texto no haya cambiado.

## SQL RDS

Contrato de referencia:

```text
sql/aws_pgvector_contract.sql
```

Esquemas exactos:

```text
docs/pgvector_place_embeddings_schema.md
docs/pgvector_post_embeddings_schema.md
```

Ese SQL debe ejecutarse una vez con un rol administrador/DBA fuera de Hugging Face. La API NLP usa solo `nlp_reader`; los jobs usan solo `nlp_writer`.

### Migracion De FastText VECTOR(300) A E5 VECTOR(384)

La guia completa de modelo, dataset, entrenamiento, PGVector y Space esta en
`docs/sentence_transformer_finetuning.md`.

Los vectores son indices derivados y no se pueden convertir entre modelos. Con el
Space y los jobs pausados, ejecuta desde pgAdmin:

```text
sql/pgadmin_migrate_sentence_transformer_384.sql
```

La migracion vacia solamente `place_embeddings` y `post_embeddings`, cambia ambas
columnas a `VECTOR(384)` y reconstruye funciones, indices y permisos. Despues repuebla
ambas tablas con los scripts de Colab y ejecuta:

```text
sql/verify_sentence_transformer_embeddings.sql
```

No repitas la migracion una vez que las columnas sean `VECTOR(384)`.

## Ranking Semantico E5 Y Llama Via Groq

`/places/search` y `/places/recommendations` generan embeddings contextuales con un
encoder E5. Las consultas usan el prefijo `query:` y los documentos almacenados usan
`passage:`. Ambos vectores se normalizan y PGVector ordena por similitud coseno.

Las requests y responses HTTP no cambian. Las metricas describen el motor
`dense_semantic_embeddings`, similitud coseno y dimension 384. `match_quality` usa
`SEMANTIC_NO_MATCH_THRESHOLD` y `SEMANTIC_RELEVANCE_THRESHOLD`; estos umbrales deben
calibrarse nuevamente cuando se publica una version afinada.

Ambos endpoints aceptan filtro geografico mediante `lat`, `lng` y `radius` en metros. Cuando se proporcionan coordenadas, el servicio NLP consulta `GET /api/v1/places/nearby` en la API principal y limita pgvector a los IDs devueltos. Las coordenadas siguen perteneciendo a la API principal; no es necesario guardarlas en pgvector, truncar tablas ni regenerar embeddings.

```json
{
  "query": "un lugar tranquilo para cenar cerca de mi",
  "lat": 16.7531,
  "lng": -93.1156,
  "radius": 10000,
  "limit": 5
}
```

El bloque `metrics` tambien indica `location_filter_applied`, `nearby_place_count` y `radius_meters` para hacer visible la aplicacion del radio.

`POST /places/recommendations` no mezcla el benchmark fijo con la consulta del usuario.
Si el score maximo no supera `SEMANTIC_NO_MATCH_THRESHOLD`, envia a Llama el modo
`no_match` y devuelve `places: []`. Entre ese valor y
`SEMANTIC_RELEVANCE_THRESHOLD` usa `low_confidence`; por encima usa `confident`.
Llama solo embellece el tono y recibe exclusivamente los lugares seleccionados.

### Documento Semantico Ponderado De Lugares

Los IDs numericos de tags devueltos por la API principal se resuelven mediante el
catalogo versionado en `app/modules/places/infrastructure/place_tag_catalog.json`.
El documento que se envia al encoder contiene exclusivamente señales semanticas y
aplica estos pesos mediante repeticion antes de generar el embedding contextual:

```text
tags x6, category x4, description x3, name x1
```

Las categorias generales se expanden con terminos de intencion en espanol. Direccion,
ciudad, estado, `source`, precio e IDs desconocidos permanecen fuera del embedding;
siguen disponibles como metadatos o filtros cuando corresponde. La version interna
`weighted-tags-v2` forma parte del hash del documento, por lo que ejecutar nuevamente
`initial_load_place_embeddings` actualiza todas las filas por `UPSERT` sin truncar la
tabla ni cambiar `VECTOR(384)`.

`GET` o `POST /places/search/metrics?k=5` conserva un benchmark offline separado llamado `built_in_places_v3_bm25`. Contiene doce lugares controlados, diez consultas y qrels graduados para calcular honestamente `Precision@k`, `Recall@k`, `MRR`, `MAP` y `nDCG@k`. Estas metricas requieren juicios de relevancia y por eso no se presentan como si midieran una consulta arbitraria de produccion.

La respuesta incluye `metric_definitions` con etiquetas y descripciones claras, y `recommended_metric` con `nDCG@k` como metrica principal sugerida para la app movil. `nDCG@k` es apropiada para recomendaciones de lugares porque considera el orden y permite relevancia graduada.

```http
GET /places/search/metrics?k=5
```

Para actualizar funciones o permisos sin cambiar nuevamente la dimension, vuelve a
ejecutar `sql/aws_pgvector_contract.sql`. No repitas la migracion destructiva una vez
que las columnas ya sean `VECTOR(384)`.

Groq/Llama se usa en `/places/recommendations` y `/places/chat` para redactar una respuesta conversacional. No decide que lugares recomendar, no hace busqueda y no inventa lugares.

El arreglo estructurado `places` viene desde RDS/pgvector mediante embeddings, filtros y ranking. La app debe renderizar cards desde ese arreglo, no parseando texto libre del LLM.

## Tests

```powershell
pytest
```
