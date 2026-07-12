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
POST /internal/posts/feed/rank
POST /internal/posts/clusters/runs
GET  /internal/posts/clusters/status
GET  /internal/posts/clusters/runs/{run_id}
POST /internal/posts/clusters/runs/{run_id}/activate
POST /search
```

## Sync Jobs

Probar sin escribir:

```powershell
python -m app.jobs.sync_place_embeddings --dry-run --max-pages 1
python -m app.jobs.sync_post_embeddings --mode snapshot --max-pages 1
python -m app.jobs.sync_search_embeddings --resource all --dry-run --max-pages 1
```

Primera carga:

```powershell
python -m app.jobs.initial_load_place_embeddings
python -m app.jobs.initial_load_post_embeddings
python -m app.jobs.initial_load_search_embeddings --resource all
```

Sincronizaciones posteriores:

```powershell
python -m app.jobs.sync_place_embeddings
python -m app.jobs.sync_post_embeddings --mode incremental
python -m app.jobs.sync_feed_interactions
python -m app.jobs.rebuild_user_interest_profiles
python -m app.jobs.train_post_clusters
python -m app.jobs.sync_search_embeddings --resource all
```

Los jobs calculan un `content_hash` versionado con el contenido, modelo, version y
dimension. Un cambio de modelo fuerza la regeneracion aunque el texto no haya cambiado.

La sincronizacion de users, clubs, groups y events consume exclusivamente los snapshots
internos de la API principal:

```text
/api/v1/internal/search/users/snapshot
/api/v1/internal/search/clubs/snapshot
/api/v1/internal/search/groups/snapshot
/api/v1/internal/search/events/snapshot
```

Estos clientes requieren `MAIN_API_INTERNAL_TOKEN`; nunca usan un JWT de usuario ni
hacen fallback a `MAIN_API_AUTH_TOKEN`.

## Busqueda Global Hibrida

La app movil debe llamar `POST /api/v1/search` en la API principal con su token de
sesion. Go valida permisos y filtros de negocio, llama al endpoint interno
`POST /internal/search/candidates` de NLP, hidrata los IDs y devuelve los recursos
completos. NLP solo calcula candidatos y puntajes; no es la fuente de verdad ni devuelve
datos de presentacion.

El query genera un solo embedding FastText. Cada proveedor combina similitud coseno con
full-text search de PostgreSQL mediante Reciprocal Rank Fusion. Los resultados que no
superan el umbral semantico ni el lexical se descartan: no se agregan recursos de relleno.
Los eventos se descartan antes del ranking cuando
`start_time + duration_minutes <= as_of`.

`POST /search` permanece como endpoint publico de compatibilidad temporal. Puede
deshabilitarse con `PUBLIC_GLOBAL_SEARCH_ENABLED=false` cuando la app movil ya use solo
la API principal.

```json
{
  "query": "club universitario de ajedrez",
  "resource_types": ["clubs", "groups", "events", "users"],
  "per_type_limit": 5,
  "top_limit": 10,
  "requester_id": "b83ab97e-91a4-4f69-b102-b27c6092e9cb"
}
```

La API principal obtiene la identidad desde el token de login; la app no debe enviar
`requester_id` como autoridad. El endpoint interno de NLP exige
`Authorization: Bearer <NLP_SERVICE_TOKEN>` y devuelve exclusivamente IDs y puntajes.

El indice de usuarios excluye correo, fecha de nacimiento, genero y ubicacion actual. Los
IDs de tags de eventos se conservan como metadatos; para aportar significado semantico la
API principal debe enviar tambien sus nombres.

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

El feed requiere, en este orden para una BD ya existente:

```text
sql/migrate_post_feed_v1.sql
sql/verify_post_feed_v1.sql
sql/migrate_post_feed_v2.sql
sql/verify_post_feed_v2.sql
```

`sql/rollback_post_feed_v1.sql` es el rollback destructivo de emergencia. No se debe
ejecutar ninguna de estas migraciones desde la API ni desde un job.

La busqueda coordinada por la API principal requiere, para una BD existente:

```text
sql/migrations/20260712_01_global_search_candidate_filters.sql
sql/verify_global_search_candidate_filters.sql
```

La migracion reemplaza de forma transaccional `search_resource_embeddings` y agrega dos
helpers de parseo seguro para timestamps y duraciones. No crea ni modifica tablas,
columnas o indices. El segundo archivo es de solo lectura, falla con una excepcion si
detecta deriva y debe ejecutarse despues para comprobar el contrato.

`/ready` exige la firma y los marcadores V2 de search; una funcion heredada ya no puede
declarar el servicio listo. Las conexiones y consultas pgvector usan
`REQUEST_TIMEOUT_SECONDS`, y las rutas tienen un limite local configurable con
`RATE_LIMIT_REQUESTS_PER_WINDOW`, `INTERNAL_RATE_LIMIT_REQUESTS_PER_WINDOW` y
`RATE_LIMIT_WINDOW_SECONDS`.

Para una BD vacía o una BD existente que ya use `VECTOR(300)`, puede ejecutarse
`sql/new_pgvector_schema.sql` desde pgAdmin. El archivo es convergente e incluye
el contrato base, feed V1, correcciones V2, propietarios y permisos. No convierte
`VECTOR(16)`; esa conversión sigue usando la migración FastText separada.

Los perfiles no se actualizan dentro del request móvil. Configura un scheduler
(cron, EventBridge o worker) con estos comandos, en orden:

```powershell
python -m app.jobs.sync_post_embeddings --mode incremental
python -m app.jobs.sync_feed_interactions
```

Una frecuencia inicial razonable es cada minuto. El entrenamiento de clusters se
ejecuta aparte con `python -m app.jobs.train_post_clusters`; con pocos posts ajusta
`KMEANS_MIN_POSTS`, `KMEANS_MIN_K` y `KMEANS_MIN_CLUSTER_SIZE` sin usar `k >= n`.

El endpoint `POST /internal/posts/clusters/runs` agenda el entrenamiento en background
y responde `202`. Para operacion normal se recomienda el job
`python -m app.jobs.train_post_clusters`. Si `KMEANS_AUTO_ACTIVATE=false`, activa el run
validado con `POST /internal/posts/clusters/runs/{run_id}/activate`.

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

### Documento Semantico Ponderado De Lugares

Los IDs numericos de tags devueltos por la API principal se resuelven mediante el
catalogo versionado en `app/modules/places/infrastructure/place_tag_catalog.json`.
El documento que se envia a FastText contiene exclusivamente señales semanticas y
aplica estos pesos mediante repeticion antes del promedio de embeddings:

```text
tags x6, category x4, description x3, name x1
```

Las categorias generales se expanden con terminos de intencion en espanol. Direccion,
ciudad, estado, `source`, precio e IDs desconocidos permanecen fuera del embedding;
siguen disponibles como metadatos o filtros cuando corresponde. La version interna
`weighted-tags-v2` forma parte del hash del documento, por lo que ejecutar nuevamente
`initial_load_place_embeddings` actualiza todas las filas por `UPSERT` sin truncar la
tabla ni cambiar `VECTOR(300)`.

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
