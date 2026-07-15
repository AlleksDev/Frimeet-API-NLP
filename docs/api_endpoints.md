# API HTTP de Frimeet NLP

Documentacion del contrato HTTP publicado por el servicio NLP de Frimeet.

## URLs base

Produccion en Hugging Face Spaces:

```text
https://alleksdev-frimeet-api-nlp.hf.space
```

Desarrollo local:

```text
http://localhost:8080
```

Documentacion interactiva:

- Swagger UI: `GET /docs`
- ReDoc: `GET /redoc`
- Especificacion OpenAPI: `GET /openapi.json`

Todas las peticiones y respuestas con body utilizan `application/json`.

## Resumen de endpoints

| Metodo | Ruta | Proposito | Autenticacion |
| --- | --- | --- | --- |
| `GET` | `/` | Informacion basica del servicio | No |
| `GET` | `/health` | Comprobar que el proceso responde | No |
| `GET` | `/ready` | Comprobar dependencias y contrato pgvector | No |
| `GET`, `POST` | `/places/search/metrics` | Evaluar el buscador con el benchmark integrado | No |
| `POST` | `/places/search` | Buscar lugares semanticamente | No |
| `POST` | `/places/recommendations` | Recomendar lugares y redactar una respuesta | No |
| `POST` | `/places/chat` | Conversacion orientada a lugares | No |
| `POST` | `/internal/places/chat` | Interpretar chat y devolver candidatos tecnicos | Servicio |
| `POST` | `/posts/recommendations` | Recomendar publicaciones | No |
| `POST` | `/internal/posts/feed/rank` | Ordenar allowlist de posts | Servicio |
| `POST` | `/internal/posts/clusters/runs` | Agendar entrenamiento de clusters | Servicio |
| `GET` | `/internal/posts/clusters/status` | Estado del run activo | Servicio |
| `GET` | `/internal/posts/clusters/runs/{id}` | Detalle de un run de clusters | Servicio |
| `POST` | `/internal/posts/clusters/runs/{id}/activate` | Activar/rollback de run | Servicio |
| `POST` | `/search` | Busqueda global sobre todos los recursos | Condicional |
| `POST` | `/internal/search/candidates` | Candidatos tecnicos para la API principal | Servicio |

La autenticacion de `/search` solo es obligatoria cuando el body contiene
`requester_id`. En ese caso se envia:

```http
Authorization: Bearer <SEARCH_INTERNAL_TOKEN>
```

El token debe ser enviado por la API principal, no directamente por la aplicacion movil.

## 1. Endpoints del sistema

### `GET /`

Devuelve informacion basica y enlaces del servicio.

```bash
curl "https://alleksdev-frimeet-api-nlp.hf.space/"
```

Respuesta `200 OK`:

```json
{
  "service": "frimeet-api-nlp",
  "status": "ok",
  "docs": "/docs",
  "health": "/health",
  "ready": "/ready"
}
```

### `GET /health`

Indica que el proceso HTTP esta vivo. No comprueba PostgreSQL ni las funciones de
pgvector.

Respuesta `200 OK`:

```json
{
  "status": "ok"
}
```

### `GET /ready`

Comprueba que el servicio puede utilizar su almacenamiento vectorial. Cuando
`VECTOR_STORE_PROVIDER=aws_pgvector`, valida:

- disponibilidad de la extension `vector`;
- existencia y permiso de ejecucion de `match_places(...)`;
- existencia y permiso de ejecucion de `match_posts(...)`;
- existencia y permiso de ejecucion de `search_resource_embeddings(...)`.
- presencia de los filtros V2 (`event_active_at`, umbrales y parseo temporal seguro).

Respuesta correcta: `200 OK` con `status: "ready"`.

Si una dependencia o funcion SQL no esta disponible, responde `503 Service Unavailable`
con `status: "not_ready"` y el detalle en `dependencies.vector_store.contract`.

## 2. Lugares

### Campos comunes de busqueda de lugares

`POST /places/search` y `POST /places/recommendations` utilizan el mismo body base.

| Campo | Tipo | Obligatorio | Default | Restricciones |
| --- | --- | --- | --- | --- |
| `query` | `string` | Si | - | 1 a 500 caracteres |
| `city` | `string \| null` | No | `null` | Maximo 80 caracteres |
| `state` | `string \| null` | No | `null` | Maximo 80 caracteres |
| `filters` | `object` | No | `{}` | Filtros adicionales |
| `limit` | `integer` | No | `10` | Entre 1 y 20 |
| `lat` | `number \| null` | No | `null` | Entre -90 y 90 |
| `lng` | `number \| null` | No | `null` | Entre -180 y 180 |
| `radius` | `integer` | No | `5000` | Entre 1 y 50000 metros |

`lat` y `lng` deben enviarse juntos. Si se omite cualquiera de los dos, la API responde
`422 Unprocessable Entity`.

Campos de `filters`:

| Campo | Tipo | Default |
| --- | --- | --- |
| `city` | `string \| null` | `null` |
| `state` | `string \| null` | `null` |
| `category` | `string \| null` | `null` |
| `price_range` | `string \| null` | `null` |
| `is_active` | `boolean \| null` | `true` |
| `occasion` | `string \| null` | `null` |

Si `city` o `state` aparecen tanto en el nivel principal como dentro de `filters`, tiene
prioridad el valor del nivel principal.

### `POST /places/search`

Realiza recuperacion semantica de lugares mediante FastText y pgvector. Llama a la API
principal para obtener un allowlist de lugares cercanos cuando se incluyen coordenadas.

No utiliza Llama/Groq para elegir resultados.

Ejemplo:

```bash
curl -X POST "https://alleksdev-frimeet-api-nlp.hf.space/places/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "cafeteria tranquila para estudiar",
    "city": "Tuxtla Gutierrez",
    "filters": {
      "category": "cafeteria",
      "is_active": true
    },
    "limit": 5
  }'
```

Ejemplo con filtro geografico:

```json
{
  "query": "un lugar tranquilo para cenar",
  "lat": 16.7531,
  "lng": -93.1156,
  "radius": 10000,
  "limit": 5
}
```

Respuesta `200 OK`:

```json
{
  "query": "cafeteria tranquila para estudiar",
  "places": [
    {
      "id": "place-id",
      "name": "Nombre del lugar",
      "score": 0.7312,
      "category": "cafeteria",
      "city": "Tuxtla Gutierrez",
      "state": "Chiapas",
      "metadata": {}
    }
  ],
  "metrics": {
    "engine": "fasttext_mean_embeddings",
    "candidate_retrieval": "pgvector",
    "score_metric": "cosine_similarity",
    "field_weights": {
      "tags": 6,
      "category": 4,
      "description": 3,
      "name": 1
    },
    "ranking_parameters": {
      "dimension": 300.0
    },
    "relevance_threshold": 0.5,
    "match_quality": "confident",
    "query_token_count": 4,
    "matched_query_token_count": 4,
    "query_coverage": 1.0,
    "scope": "request",
    "ground_truth_available": false,
    "candidate_count": 5,
    "returned_count": 5,
    "nonzero_score_count": 5,
    "min_score": 0.42,
    "max_score": 0.7312,
    "mean_score": 0.57,
    "location_filter_applied": false,
    "nearby_place_count": null,
    "radius_meters": null
  }
}
```

`match_quality` puede ser `confident`, `low_confidence` o `no_match`.

### `POST /places/recommendations`

Busca lugares con el mismo flujo de `/places/search` y despues utiliza Llama/Groq para
redactar un mensaje conversacional. El LLM solo recibe los lugares recuperados; no elige
lugares adicionales ni puede inventar IDs.

Body: igual al de `POST /places/search`.

```bash
curl -X POST "https://alleksdev-frimeet-api-nlp.hf.space/places/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "quiero una cena romantica",
    "city": "Tuxtla Gutierrez",
    "limit": 5
  }'
```

Respuesta `200 OK`:

```json
{
  "query": "quiero una cena romantica",
  "message": "Encontre algunas opciones que pueden interesarte.",
  "places": [],
  "metrics": {
    "engine": "fasttext_mean_embeddings",
    "candidate_retrieval": "pgvector",
    "score_metric": "cosine_similarity",
    "field_weights": {
      "tags": 6,
      "category": 4,
      "description": 3,
      "name": 1
    },
    "ranking_parameters": {
      "dimension": 300.0
    },
    "relevance_threshold": 0.5,
    "match_quality": "confident",
    "query_token_count": 4,
    "matched_query_token_count": 4,
    "query_coverage": 1.0,
    "scope": "request",
    "ground_truth_available": false,
    "candidate_count": 5,
    "returned_count": 5,
    "nonzero_score_count": 5,
    "min_score": 0.42,
    "max_score": 0.7312,
    "mean_score": 0.57,
    "location_filter_applied": false,
    "nearby_place_count": null,
    "radius_meters": null
  },
  "metadata": {
    "strategy": "pgvector_candidates_plus_fasttext_mean_embeddings",
    "ranking": "fasttext_mean_embeddings",
    "response_mode": "confident",
    "relevance_threshold": 0.5,
    "llm_provider": "groq",
    "llm_model": "llama-3.1-8b-instant",
    "used_llm": true,
    "guard_reason": null,
    "places_used_as_context": [],
    "timestamp": "2026-07-04T12:00:00+00:00"
  }
}
```

`metrics` tiene el mismo esquema completo mostrado en `/places/search`.

### `POST /places/chat`

Endpoint conversacional orientado a lugares.

Body:

| Campo | Tipo | Obligatorio | Default | Restricciones |
| --- | --- | --- | --- | --- |
| `message` | `string` | Si | - | 1 a 1000 caracteres |
| `city` | `string \| null` | No | `null` | Maximo 80 caracteres |
| `state` | `string \| null` | No | `null` | Maximo 80 caracteres |
| `filters` | `object` | No | `{}` | Mismo esquema de filtros de lugares |
| `limit` | `integer` | No | `5` | Entre 1 y 8 |

Ejemplo:

```json
{
  "message": "Quiero salir con amigos a un lugar con musica",
  "city": "Tuxtla Gutierrez",
  "limit": 5
}
```

Respuesta:

```json
{
  "response_id": "uuid-de-respuesta",
  "nlp_trace_id": "uuid-de-traza",
  "message": "Estas opciones pueden funcionar para tu salida.",
  "places": [],
  "metadata": {}
}
```

### `GET|POST /places/search/metrics`

Ejecuta el benchmark offline integrado para evaluar la calidad del buscador de lugares.
No evalua una consulta enviada por el usuario y no acepta body.

Query parameter:

| Parametro | Tipo | Default | Restricciones |
| --- | --- | --- | --- |
| `k` | `integer` | `5` | Entre 1 y 20 |

Ambos metodos son equivalentes:

```bash
curl "https://alleksdev-frimeet-api-nlp.hf.space/places/search/metrics?k=5"
```

La respuesta incluye:

- `engine`, `benchmark` y `qrels_source`;
- `precision_at_k`, `recall_at_k`, `mrr`, `map` y `ndcg_at_k` agregados;
- metricas individuales de cada consulta del benchmark;
- definiciones de las metricas;
- `recommended_metric`, actualmente nDCG@k.

### `POST /internal/places/chat`

Endpoint V2 consumido exclusivamente por la API principal. La app movil nunca debe
llamarlo directamente.

```http
Authorization: Bearer <NLP_SERVICE_TOKEN>
Content-Type: application/json
```

Request minimo:

```json
{
  "conversation_id": "cb3456ef-598e-49d9-9bf9-b2ba99055ad7",
  "turn": 1,
  "message": "recomiendame una cafeteria cerca del Parque Central",
  "state": {},
  "user_location": {"lat": 16.7531, "lng": -93.1156},
  "candidate_limit": 30,
  "result_limit": 5
}
```

La respuesta devuelve `action`, mensaje, `state_patch`, `location_directive`, IDs y
scores tecnicos. No devuelve cards, coordenadas ni metadata privada. `clarification` y
`no_match` siempre tienen `candidates=[]`. Go debe fusionar el patch de forma atomica,
hidratar los IDs, revalidar el anchor y aplicar distancia/PostGIS antes de responder a
la app.

`state` puede incluir `target_category`, `hard_filters`, `soft_preferences`,
`exclusions`, `reference`, `explicit_target_location` y `pending_clarification`. Si
`taxonomy_version` no coincide con la version desplegada, el endpoint responde `409`.

Ejemplo de directiva sin ubicacion explicita:

```json
{
  "source": "user_current",
  "scope": "user_current_location",
  "anchor_place_id": null,
  "anchor_text": null,
  "radius_meters": null,
  "strict_radius": false
}
```

Para `cafeterias como la de Hello Kitty cerca del Parque Central`, NLP no elige en
silencio entre usar el parque como zona de resultados o como ayuda para identificar la
referencia: devuelve `action=clarification`, `unresolved=["location_scope"]` y conserva
el contexto no ambiguo en `state_patch`.

No es obligatorio escribir una categoria literal. El parser aplica defaults de alta
precision para actividades inequívocas: `quiero comer algo` se interpreta como
`target_category=restaurant`, mientras que una frase abierta como `quiero salir` aun
solicita aclaracion. Si el mensaje tambien contiene una categoria explicita, esta tiene
prioridad sobre el default inferido.

## 3. Publicaciones

### `POST /posts/recommendations`

Recupera publicaciones relacionadas semanticamente con una consulta.

Body:

| Campo | Tipo | Obligatorio | Default | Restricciones |
| --- | --- | --- | --- | --- |
| `query` | `string` | Si | - | 1 a 500 caracteres |
| `city` | `string \| null` | No | `null` | Maximo 80 caracteres |
| `limit` | `integer` | No | `10` | Entre 1 y 20 |

Ejemplo:

```bash
curl -X POST "https://alleksdev-frimeet-api-nlp.hf.space/posts/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "personas que quieran jugar futbol",
    "city": "Tuxtla Gutierrez",
    "limit": 10
  }'
```

Respuesta:

```json
{
  "query": "personas que quieran jugar futbol",
  "posts": [
    {
      "id": "post-id",
      "title": "Partido de futbol este sabado",
      "score": 0.69,
      "city": "Tuxtla Gutierrez",
      "tags": ["futbol", "deporte"],
      "metadata": {}
    }
  ],
  "metadata": {}
}
```

### `POST /internal/posts/feed/rank`

Endpoint servicio-a-servicio. Requiere:

```http
Authorization: Bearer <NLP_SERVICE_TOKEN>
```

La API Go envia exclusivamente candidatos ya autorizados. NLP nunca introduce IDs
externos y devuelve también candidatos sin embedding con score técnico de cold start.

```json
{
  "user_id": "uuid",
  "candidate_posts": [
    {
      "post_id": "uuid",
      "author_id": "uuid",
      "created_at": "2026-07-05T12:00:00Z",
      "social_affinity": 1.0,
      "engagement_score": 0.4,
      "author_affinity": 0.0
    }
  ],
  "snapshot_at": "2026-07-05T12:01:00Z",
  "result_limit": 500
}
```

Respuesta:

```json
{
  "items": [
    {
      "post_id": "uuid",
      "score": 0.87,
      "cluster_id": 12
    }
  ],
  "ranking_version": "feed-v1",
  "cluster_run_id": "uuid",
  "cold_start": false,
  "missing_embedding_count": 0,
  "diversity_relaxations": 0,
  "duplicate_penalized_count": 0
}
```

### Operación interna de clusters

```http
POST /internal/posts/clusters/runs
GET  /internal/posts/clusters/status
GET  /internal/posts/clusters/runs/{run_id}
POST /internal/posts/clusters/runs/{run_id}/activate
Authorization: Bearer <NLP_SERVICE_TOKEN>
```

`POST /internal/posts/clusters/runs` agenda el entrenamiento y responde `202 Accepted`:

```json
{
  "status": "scheduled",
  "message": "entrenamiento enviado al worker en background"
}
```

`GET /internal/posts/clusters/runs/{run_id}` devuelve estado, metricas y error si fallo.
El entrenamiento normal debe ejecutarse con `python -m app.jobs.train_post_clusters`;
los endpoints son controles administrativos. Si `KMEANS_AUTO_ACTIVATE=false`, primero
queda en `validated` y despues se activa con el endpoint de activate.

## 4. Busqueda global

### `POST /search`

Busca en paralelo sobre:

- `places`;
- `posts`;
- `users`;
- `clubs`;
- `groups`;
- `events`.

Calcula una sola representacion FastText de la consulta. Para `places` utiliza el mismo
motor que `/places/recommendations`: similitud coseno mediante `match_places(...)`, sin
fusion lexical ni RRF. Para `posts`, `users`, `clubs`, `groups` y `events` combina
busqueda semantica pgvector con full-text search de PostgreSQL mediante Reciprocal Rank
Fusion (RRF).

No recibe query parameters. Toda la entrada se envia en el body.

#### Body

| Campo | Tipo | Obligatorio | Default | Restricciones |
| --- | --- | --- | --- | --- |
| `query` | `string` | Si | - | 1 a 500 caracteres |
| `resource_types` | `string[] \| null` | No | Todos | Solo tipos enumerados |
| `per_type_limit` | `integer` | No | `5` | Entre 1 y 20 |
| `top_limit` | `integer` | No | `10` | Entre 1 y 50 |
| `requester_id` | `UUID \| null` | No | `null` | Activa busqueda privada |
| `cursors` | `object` | No | `{}` | Cursor opaco por tipo de recurso |
| `location` | `object \| null` | No | `null` | Ubicacion y radio del usuario |
| `filters` | `object` | No | `{}` | Filtros tipados por recurso |

No se aceptan campos adicionales.

En la primera solicitud se omite `cursors`. La respuesta incluye un bloque
`pagination` independiente para cada recurso consultado. Cuando `has_more` es `true`,
`next_cursor` se envia en la siguiente solicitud usando la misma consulta.

#### Ubicacion

`location` consulta una sola vez `GET /api/v1/places/nearby` en la API principal. Las
coordenadas siguen siendo propiedad de la API principal y no se almacenan en pgvector.

| Campo | Tipo | Obligatorio | Default | Restricciones |
| --- | --- | --- | --- | --- |
| `lat` | `number` | Si | - | Entre -90 y 90 |
| `lng` | `number` | Si | - | Entre -180 y 180 |
| `radius` | `integer` | No | `5000` | Entre 1 y 50000 metros |
| `mode` | `string` | No | `prioritize` | `prioritize` o `strict` |

- `prioritize`: aumenta solamente el ranking interno de lugares cercanos, sin modificar
  su `score` semantico.
- `strict`: excluye lugares, clubs presenciales y eventos cuyo `place_id` no este dentro
  del radio.
- La ubicacion aplica a `places`, `clubs` y `events`. No se inventa proximidad para
  usuarios, posts o grupos porque esos indices no contienen una relacion geografica
  verificada.

#### Filtros

Todos los campos son opcionales. Cada filtro se aplica solamente a los recursos que
poseen esa señal.

| Campo | Recursos | Tipo | Descripcion |
| --- | --- | --- | --- |
| `city` | places, posts | `string` | Coincidencia exacta sin distinguir mayusculas |
| `state` | places, posts | `string` | Estado o region |
| `categories` | places, clubs | `string[]` | Una o varias categorias admitidas |
| `price_ranges` | places | `string[]` | Rangos como `$`, `$$` o `$$$` |
| `tags` | places, posts, events | `string[]` | Debe coincidir al menos un tag |
| `published_from` | posts | `datetime` | Fecha minima de publicacion ISO 8601 |
| `published_to` | posts | `datetime` | Fecha maxima de publicacion ISO 8601 |
| `event_from` | events | `datetime` | Inicio minimo del evento |
| `event_to` | events | `datetime` | Inicio maximo del evento |
| `club_mode` | clubs | `string` | `online` o `in_person` |
| `user_roles` | users | `string[]` | Roles admitidos, por ejemplo `cliente` |

Ejemplo completo:

```json
{
  "query": "actividad para conocer personas",
  "resource_types": ["places", "clubs", "events", "posts"],
  "per_type_limit": 5,
  "top_limit": 12,
  "location": {
    "lat": 16.7531,
    "lng": -93.1156,
    "radius": 10000,
    "mode": "prioritize"
  },
  "filters": {
    "city": "Tuxtla Gutierrez",
    "categories": ["cafe", "community"],
    "price_ranges": ["$", "$$"],
    "tags": ["musica", "cultura"],
    "published_from": "2026-07-01T00:00:00Z",
    "event_from": "2026-07-04T00:00:00Z",
    "event_to": "2026-08-04T23:59:59Z",
    "club_mode": "in_person"
  }
}
```

#### Busqueda publica

No requiere `Authorization`.

```bash
curl -X POST "https://alleksdev-frimeet-api-nlp.hf.space/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ajedrez universitario",
    "resource_types": ["clubs", "events", "users"],
    "per_type_limit": 5,
    "top_limit": 10
  }'
```

#### Busqueda con contexto de usuario

Cuando se incluye `requester_id`, la API principal debe agregar el Bearer token interno:

```bash
curl -X POST "https://alleksdev-frimeet-api-nlp.hf.space/search" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $SEARCH_INTERNAL_TOKEN" \
  -d '{
    "query": "amigos de la universidad",
    "resource_types": ["groups"],
    "requester_id": "00000000-0000-0000-0000-000000000001",
    "per_type_limit": 5,
    "top_limit": 10
  }'
```

Si falta el token, es incorrecto o no esta configurado en el servicio, responde
`403 Forbidden`.

`requester_id` nunca debe confiarse directamente desde la app movil. La API principal
debe obtenerlo de la sesion autenticada y construir la llamada interna.

#### Respuesta

```json
{
  "query": "ajedrez universitario",
  "normalized_query": "ajedrez universitario",
  "top_results": [
    {
      "id": "club-id",
      "resource_type": "clubs",
      "title": "Club de Ajedrez Universitario",
      "subtitle": "ajedrez",
      "score": 0.92,
      "semantic_score": 0.81,
      "lexical_score": 0.38,
      "is_nearby": true,
      "proximity_boost": 0.12,
      "metadata": {}
    }
  ],
  "sections": {
    "clubs": [
      {
        "id": "club-id",
        "resource_type": "clubs",
        "title": "Club de Ajedrez Universitario",
        "subtitle": "ajedrez",
        "score": 0.92,
        "semantic_score": 0.81,
        "lexical_score": 0.38,
        "is_nearby": true,
        "proximity_boost": 0.12,
        "metadata": {}
      }
    ],
    "events": [],
    "users": []
  },
  "pagination": {
    "clubs": {
      "page_size": 5,
      "returned_count": 1,
      "has_more": false,
      "next_cursor": null
    },
    "events": {
      "page_size": 5,
      "returned_count": 0,
      "has_more": false,
      "next_cursor": null
    },
    "users": {
      "page_size": 5,
      "returned_count": 0,
      "has_more": false,
      "next_cursor": null
    }
  },
  "metadata": {
    "strategy": "parallel_hybrid_fasttext_full_text_rrf",
    "queried_resources": ["clubs", "events", "users"],
    "failed_resources": {},
    "embedding_computed_once": true
  }
}
```

`top_results` es una seleccion global diversificada. `sections` conserva los resultados
separados por tipo de recurso.

#### Solicitar la siguiente pagina

Cada recurso se pagina de manera independiente. Para cargar mas lugares, la interfaz
debe reutilizar exactamente `query`, declarar `resource_types: ["places"]` y enviar el
cursor recibido en `pagination.places.next_cursor`:

```json
{
  "query": "cafeteria tranquila",
  "resource_types": ["places"],
  "per_type_limit": 5,
  "top_limit": 5,
  "cursors": {
    "places": "CURSOR_DEVUELTO_POR_LA_PAGINA_ANTERIOR"
  }
}
```

La respuesta de cada seccion contiene:

| Campo | Descripcion |
| --- | --- |
| `page_size` | Limite solicitado para esa pagina |
| `returned_count` | Cantidad realmente devuelta |
| `has_more` | Indica si existe al menos otra pagina |
| `next_cursor` | Cursor opaco de continuacion o `null` si termino |

Reglas de los cursores:

- estan asociados al tipo de recurso, consulta, filtros, ubicacion, usuario y tamano de pagina;
- no deben interpretarse ni construirse en la app cliente;
- no pueden reutilizarse con otra consulta o con otro recurso;
- cuando se envia `cursors`, `resource_types` es obligatorio y debe contener sus claves;
- se pueden pedir varias continuaciones en una llamada enviando un cursor por recurso.

El contrato SQL utiliza el ID externo como desempate estable cuando dos resultados tienen
el mismo puntaje. Esto evita cambios arbitrarios de orden entre paginas consecutivas.

Si falla un proveedor individual, los demas pueden responder normalmente. El recurso
fallido aparece en `metadata.failed_resources`.

#### Privacidad

- Sin `requester_id` solo se consideran recursos activos y publicos.
- Con `requester_id`, un grupo privado puede aparecer si el usuario es creador, miembro
  agregado o invitado segun `authorized_user_ids`.
- `creator_id` y `authorized_user_ids` se utilizan internamente, pero se eliminan de la
  metadata enviada al cliente.

## 5. Errores comunes

| Estado | Significado |
| --- | --- |
| `403` | Falta o es incorrecto el Bearer token de una busqueda con `requester_id` |
| `413` | El body supera el limite configurado por `MAX_REQUEST_BODY_BYTES` |
| `422` | Body, UUID, coordenadas, limites o query parameters invalidos |
| `503` | PostgreSQL, pgvector o una funcion SQL requerida no esta disponible |

Ejemplo de error de validacion `422`:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "query"],
      "msg": "Field required"
    }
  ]
}
```

## 6. Candidatos internos de busqueda

### `POST /internal/search/candidates`

Endpoint de servicio consumido exclusivamente por la API principal. La app movil no
debe llamarlo directamente.

```http
Authorization: Bearer <NLP_SERVICE_TOKEN>
Content-Type: application/json
```

Ejemplo minimo:

```json
{
  "query": "club universitario de ajedrez",
  "resource_types": ["clubs", "groups", "events", "posts"],
  "candidate_limit_per_type": 50,
  "top_limit": 100,
  "as_of": "2026-07-12T14:00:00Z"
}
```

La respuesta contiene secciones independientes con `id`, `resource_type`, `score`,
`semantic_score` y `lexical_score`. No incluye metadata de UI ni registros hidratados.
La API principal aplica permisos y reglas de negocio, carga los recursos vigentes y
conserva el orden del ranking.

NLP aplica antes del `LIMIT` los umbrales configurados. Un candidato se acepta cuando
supera el umbral semantico **o** el lexical; si ninguno lo supera se descarta y no se
rellena la respuesta. Para eventos tambien exige que
`start_time + duration_minutes > as_of`, evitando calcular ranking sobre eventos ya
finalizados. Los cursores quedan ligados a query, recursos, filtros, ubicacion,
`as_of` y version de la politica de umbrales.

## 7. Variables relacionadas

| Variable | Uso |
| --- | --- |
| `SEARCH_INTERNAL_TOKEN` | Valida `Authorization: Bearer ...` cuando `/search` recibe `requester_id` |
| `NLP_SERVICE_TOKEN` | Autentica `/internal/posts/*`, `/internal/search/candidates` y `/internal/places/chat`; debe coincidir con Go |
| `MAIN_API_INTERNAL_TOKEN` | Autentica jobs y resolucion de anchors al consumir endpoints internos de Go |
| `MAIN_API_PLACE_ANCHOR_RESOLVE_PATH` | Ruta interna Go para resolver nombres de lugares usados como anchors o referencias |
| `PLACES_CHAT_V2_ENABLED` | Feature flag del chat interno; debe iniciar en `false` |
| `PLACES_CHAT_LLM_ENABLED` | Habilita solo la redaccion opcional; no cambia candidatos ni accion |
| `PLACES_CHAT_CANDIDATE_LIMIT` | Maximo tecnico de candidatos de contenido, entre 1 y 40 |
| `PLACES_CHAT_MIN_CONTENT_SCORE` | Umbral minimo antes de devolver candidatos |
| `PLACES_CHAT_INTENT_MIN_CONFIDENCE` | Confianza minima; por debajo se solicita aclaracion |
| `PLACES_CHAT_AMBIGUITY_DELTA` | Diferencia maxima para considerar ambiguos dos anchors |
| `PLACES_CHAT_RANKING_VERSION` | Version observable de la politica de ranking |
| `PLACES_CHAT_TAXONOMY_VERSION` | Version del parser y del estado conversacional |
| `MAX_REQUEST_BODY_BYTES` | Debe ser al menos `131072`; valor recomendado `262144` para 500 candidatos |
| `REQUEST_TIMEOUT_SECONDS` | Presupuesto maximo para search interno y timeout de conexion/consulta pgvector |
| `RATE_LIMIT_REQUESTS_PER_WINDOW` | Limite local por IP/ruta para endpoints publicos |
| `INTERNAL_RATE_LIMIT_REQUESTS_PER_WINDOW` | Limite local por IP/ruta para endpoints internos |
| `RATE_LIMIT_WINDOW_SECONDS` | Ventana del limitador local; el gateway puede agregar un limite distribuido |
| `VECTOR_STORE_PROVIDER` | Selecciona `aws_pgvector` o el proveedor mock |
| `PGVECTOR_*` | Conexion y roles de PostgreSQL/pgvector |
| `EMBEDDING_PROVIDER` | Proveedor de embeddings; produccion utiliza `fasttext` |
| `EMBEDDING_DIMENSION` | Dimension vectorial; FastText utiliza `300` |
| `GROQ_API_KEY` | Habilita Groq/Llama para redactar respuestas conversacionales |
| `MAIN_API_BASE_URL` | API principal usada para fuentes y filtros geograficos |
| `MAIN_API_USERS_SNAPSHOT_PATH` | Snapshot interno paginado de usuarios para search |
| `MAIN_API_CLUBS_SNAPSHOT_PATH` | Snapshot interno paginado de clubs para search |
| `MAIN_API_GROUPS_SNAPSHOT_PATH` | Snapshot interno paginado de grupos para search |
| `MAIN_API_EVENTS_SNAPSHOT_PATH` | Snapshot interno paginado de eventos para search |
| `MAIN_API_PLACES_NEARBY_PATH` | Endpoint que resuelve los IDs dentro del radio solicitado |
| `GLOBAL_SEARCH_NEARBY_BOOST` | Peso de priorizacion geografica; default `0.12` |
| `PUBLIC_GLOBAL_SEARCH_ENABLED` | Mantiene o deshabilita el endpoint publico heredado `POST /search` |
| `GLOBAL_SEARCH_MIN_SEMANTIC_SCORE` | Umbral semantico global; default `0.30` |
| `GLOBAL_SEARCH_MIN_LEXICAL_SCORE` | Umbral lexical global; default `0.05` |
| `GLOBAL_SEARCH_RESOURCE_THRESHOLDS_JSON` | Overrides opcionales por recurso con `semantic_min` y `lexical_min` |
| `GLOBAL_SEARCH_THRESHOLD_POLICY_VERSION` | Version estable incluida en cursores para invalidarlos al cambiar la politica |

`top_limit` se conserva en el request y en la huella del cursor para mantener el
contexto coordinado con Go. NLP devuelve candidatos por seccion; la composicion final de
`top_results` pertenece a la API principal.
