---
title: Frimeet API NLP
sdk: docker
app_port: 7860
pinned: false
---

# Frimeet API NLP

Servicio NLP independiente para busqueda semantica con FastText + pgvector, recomendaciones y redaccion conversacional con Llama via Groq.

La API principal sigue siendo la fuente de verdad de lugares, posts, usuarios, sesiones, permisos y reportes. Este servicio NLP solo trabaja con datos derivados para busqueda semantica.

## Arquitectura

```text
API principal
  |-- fuente de verdad de places/posts
  `-- consume la API NLP por REST

Hugging Face API NLP
  |-- usa credenciales nlp_reader
  |-- consulta RDS PostgreSQL + pgvector
  |-- genera el embedding FastText del query del usuario
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

MAIN_API_BASE_URL=http://52.86.8.11
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

EMBEDDING_PROVIDER=fasttext
EMBEDDING_DIMENSION=300
EMBEDDING_MODEL=facebook/fasttext-es-vectors
EMBEDDING_VERSION=common-crawl-300-v1
FASTTEXT_MODEL_PATH=.models/fasttext-es/model.bin
FASTTEXT_MODEL_REPO_ID=facebook/fasttext-es-vectors
FASTTEXT_MODEL_FILENAME=model.bin
FASTTEXT_AUTO_DOWNLOAD=true

BM25_K1=1.5
BM25_B=0.75
BM25_RELEVANCE_THRESHOLD=3.0
SEMANTIC_NO_MATCH_THRESHOLD=0.30
SEMANTIC_RELEVANCE_THRESHOLD=0.50

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

Durante el build, Docker descarga `model.bin` desde el repositorio oficial
`facebook/fasttext-es-vectors` y lo guarda en `/opt/models/fasttext-es/model.bin`.
La capa queda cacheada, por lo que un cambio normal de codigo no vuelve a descargar
el modelo de varios GB.

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

### Migracion De VECTOR(16) A FastText VECTOR(300)

La guia operativa completa esta en `docs/fasttext_deployment.md`.

Los vectores son datos derivados. Para esta migracion no se intenta convertir los
16 valores mock en 300 valores semanticos: se vacian ambas tablas y se reconstruyen
desde la API principal.

Con la API NLP y los jobs pausados, ejecuta como `nlp_owner` o administrador:

```powershell
psql "host=<host> port=5432 dbname=nlp_vectors user=<admin> sslmode=require" -f sql/migrate_fasttext_300.sql
psql "host=<host> port=5432 dbname=nlp_vectors user=<admin> sslmode=require" -f sql/aws_pgvector_contract.sql
```

Despues configura las variables FastText, despliega la nueva imagen y repuebla:

```powershell
python -m app.jobs.initial_load_place_embeddings
python -m app.jobs.initial_load_post_embeddings
psql "host=<host> port=5432 dbname=nlp_vectors user=<admin> sslmode=require" -f sql/verify_fasttext_embeddings.sql
```

La verificacion debe reportar dimension `300`, modelo
`facebook/fasttext-es-vectors` y normas cercanas a `1`.

## Ranking Semantico FastText Y Llama Via Groq

`/places/search` y `/places/recommendations` aplican el flujo de
`Lab5_Embeddings_Busqueda_Semantica.ipynb`: tokenizan el texto, obtienen los vectores
FastText de cada termino, calculan su promedio, normalizan el documento y consultan
pgvector mediante similitud coseno. FastText usa subpalabras, por lo que puede relacionar
variantes morfologicas y palabras fuera de vocabulario.

Las requests y responses HTTP no cambian. Las metricas existentes ahora describen el
motor `fasttext_mean_embeddings`, similitud coseno y dimension 300. `match_quality`
usa `SEMANTIC_NO_MATCH_THRESHOLD` y `SEMANTIC_RELEVANCE_THRESHOLD`.

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

`GET` o `POST /places/search/metrics?k=5` conserva un benchmark offline separado llamado `built_in_places_v3_bm25`. Contiene doce lugares controlados, diez consultas y qrels graduados para calcular honestamente `Precision@k`, `Recall@k`, `MRR`, `MAP` y `nDCG@k`. Estas metricas requieren juicios de relevancia y por eso no se presentan como si midieran una consulta arbitraria de produccion.

La respuesta incluye `metric_definitions` con etiquetas y descripciones claras, y `recommended_metric` con `nDCG@k` como metrica principal sugerida para la app movil. `nDCG@k` es apropiada para recomendaciones de lugares porque considera el orden y permite relevancia graduada.

```http
GET /places/search/metrics?k=5
```

Para actualizar funciones o permisos sin cambiar nuevamente la dimension, vuelve a
ejecutar `sql/aws_pgvector_contract.sql`. No repitas la migracion destructiva una vez
que las columnas ya sean `VECTOR(300)`.

Groq/Llama se usa en `/places/recommendations` y `/places/chat` para redactar una respuesta conversacional. No decide que lugares recomendar, no hace busqueda y no inventa lugares.

El arreglo estructurado `places` viene desde RDS/pgvector mediante embeddings, filtros y ranking. La app debe renderizar cards desde ese arreglo, no parseando texto libre del LLM.

## Tests

```powershell
pytest
```
