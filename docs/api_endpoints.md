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
| `POST` | `/posts/recommendations` | Recomendar publicaciones | No |
| `GET` | `/posts/clusters` | Consultar agrupaciones de publicaciones | No |
| `POST` | `/search` | Busqueda global sobre todos los recursos | Condicional |

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

### `GET /posts/clusters`

Devuelve agrupaciones de publicaciones relacionadas.

No recibe body ni query parameters.

```json
{
  "clusters": [
    {
      "id": "cluster-id",
      "label": "Deportes",
      "post_ids": ["post-1", "post-2"],
      "size": 2,
      "metadata": {}
    }
  ],
  "metadata": {}
}
```

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

No se aceptan campos adicionales.

En la primera solicitud se omite `cursors`. La respuesta incluye un bloque
`pagination` independiente para cada recurso consultado. Cuando `has_more` es `true`,
`next_cursor` se envia en la siguiente solicitud usando la misma consulta.

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

- estan asociados al tipo de recurso y a la consulta normalizada;
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

## 6. Variables relacionadas

| Variable | Uso |
| --- | --- |
| `SEARCH_INTERNAL_TOKEN` | Valida `Authorization: Bearer ...` cuando `/search` recibe `requester_id` |
| `VECTOR_STORE_PROVIDER` | Selecciona `aws_pgvector` o el proveedor mock |
| `PGVECTOR_*` | Conexion y roles de PostgreSQL/pgvector |
| `EMBEDDING_PROVIDER` | Proveedor de embeddings; produccion utiliza `fasttext` |
| `EMBEDDING_DIMENSION` | Dimension vectorial; FastText utiliza `300` |
| `GROQ_API_KEY` | Habilita Groq/Llama para redactar respuestas conversacionales |
| `MAIN_API_BASE_URL` | API principal usada para fuentes y filtros geograficos |
