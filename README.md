# Frimeet API NLP

Servicio NLP independiente para busqueda semantica, recomendaciones, ranking, embeddings y redaccion conversacional con Llama via Groq.

La API principal sigue siendo la fuente de verdad de lugares, posts, usuarios, sesiones, permisos y reportes. Este servicio NLP solo trabaja con datos derivados para busqueda semantica.

## Arquitectura

```text
API principal
  ├─ fuente de verdad de places/posts
  └─ consume la API NLP por REST

Hugging Face API NLP
  ├─ usa nlp_reader
  ├─ consulta RDS PostgreSQL + pgvector
  ├─ genera embedding solo del query del usuario
  └─ usa Groq/Llama solo para embellecer chat

Hugging Face Jobs
  ├─ usan nlp_writer
  ├─ consumen endpoints internos paginados/incrementales de la API principal
  ├─ generan embeddings por batch
  └─ hacen upsert via funciones SQL controladas

RDS PostgreSQL + pgvector
  ├─ place_embeddings
  ├─ post_embeddings
  ├─ match_places
  ├─ match_posts
  ├─ upsert_place_embedding
  └─ upsert_post_embedding
```

## Estructura

```text
app/
├── main.py
├── modules/
│   ├── places/
│   └── posts/
├── shared/
│   ├── vector_store/
│   ├── nlp/
│   ├── config/
│   ├── cache/
│   ├── errors/
│   ├── logging/
│   └── security/
└── jobs/

sql/
└── aws_pgvector_contract.sql
```

## Vector Store

Los casos de uso no dependen directamente de PostgreSQL. Dependen de puertos por modulo:

- `PlaceVectorRepository`
- `PostVectorRepository`

Implementaciones actuales:

- `MockPlaceVectorRepository` y `MockPostVectorRepository` para desarrollo/tests.
- `AwsPgvectorPlaceRepository` y `AwsPgvectorPostRepository` para RDS/Aurora PostgreSQL + pgvector.

La API NLP solo lee mediante:

- `match_places`
- `match_posts`

Los jobs escriben mediante:

- `upsert_place_embedding`
- `upsert_post_embedding`

No se usan credenciales master/admin desde Hugging Face.

## Variables De Entorno

```env
ENV=local
API_HOST=0.0.0.0
API_PORT=8080

MAIN_API_BASE_URL=http://52.86.8.11
MAIN_API_PLACES_SEARCH_PATH=/api/v1/places/search
MAIN_API_POSTS_SEARCH_PATH=/api/v1/posts/search
MAIN_API_INTERNAL_TOKEN=
MAIN_API_TIMEOUT_SECONDS=15
MAIN_API_PLACES_PAGE_LIMIT=100
MAIN_API_POSTS_PAGE_LIMIT=100
MAIN_API_PLACES_PAGINATION_MODE=page
MAIN_API_POSTS_PAGINATION_MODE=page

GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant

VECTOR_STORE_PROVIDER=mock

PGVECTOR_HOST=
PGVECTOR_PORT=5432
PGVECTOR_DATABASE=
PGVECTOR_USER=
PGVECTOR_PASSWORD=
PGVECTOR_SSL_MODE=require

EMBEDDING_DIMENSION=16
EMBEDDING_MODEL=mock-embedding
EMBEDDING_VERSION=v1

LOG_LEVEL=INFO
REQUEST_TIMEOUT_SECONDS=10
LLM_TIMEOUT_SECONDS=12
MAX_LLM_CONCURRENT_REQUESTS=4
```

Para la API NLP en Hugging Face:

```env
VECTOR_STORE_PROVIDER=aws_pgvector
PGVECTOR_USER=nlp_reader
```

Para Hugging Face Jobs:

```env
VECTOR_STORE_PROVIDER=aws_pgvector
PGVECTOR_USER=nlp_writer
```

Todas las conexiones a RDS deben usar:

```env
PGVECTOR_SSL_MODE=require
```

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

Endpoints de sistema:

```http
GET /health
GET /ready
```

Documentacion local:

```text
http://127.0.0.1:8080/docs
```

## Endpoints

```http
POST /places/search
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

Sincronizar hacia RDS:

```powershell
python -m app.jobs.sync_place_embeddings
python -m app.jobs.sync_post_embeddings
```

Flujo de `sync_place_embeddings.py`:

1. Lee configuracion.
2. Consulta la API principal con paginacion/cursor.
3. Construye texto para embedding por lugar.
4. Calcula `content_hash`.
5. Consulta hashes existentes y omite registros sin cambios.
6. Genera embeddings por batch.
7. Llama `upsert_place_embedding`.
8. Registra procesados, omitidos, upserts y errores.

Flujo de `sync_post_embeddings.py`:

1. Lee configuracion.
2. Consulta la API principal con paginacion/cursor.
3. Construye texto para embedding por post.
4. Calcula `content_hash`.
5. Consulta hashes existentes y omite registros sin cambios.
6. Genera embeddings por batch.
7. Llama `upsert_post_embedding`.
8. Registra metricas basicas.

Si un lugar o post llega como `is_active=false`, el job actualiza ese estado derivado en RDS.

## SQL RDS

Contrato de referencia:

```text
sql/aws_pgvector_contract.sql
```

Ese archivo debe ejecutarse una vez con un rol administrador/DBA fuera de Hugging Face.

La API NLP usa solo `nlp_reader`.

Los jobs usan solo `nlp_writer`.

## Llama Via Groq

Groq/Llama solo se usa en `/places/chat` para redactar una respuesta conversacional. No decide que lugares recomendar, no hace busqueda y no inventa lugares.

El arreglo estructurado `places` viene desde RDS/pgvector mediante embeddings, filtros y ranking. La app debe renderizar cards desde ese arreglo, no parseando texto libre del LLM.

## Tests

```powershell
pytest
```
