# PgVector Place Embeddings Schema

Este es el contrato actual para guardar lugares derivados en RDS PostgreSQL + pgvector.

La API principal sigue siendo la fuente de verdad. Esta tabla contiene datos derivados para busqueda semantica.

## Tabla

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS place_embeddings (
    external_id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(300) NOT NULL,
    content_hash TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

`VECTOR(300)` es el indice legacy FastText. El chat de lugares puede usar en paralelo
`place_embeddings_semantic_v1` con `VECTOR(768)`; se crea con
`sql/migrations/20260716_02_places_semantic_v1.sql`.

Antes de generar el vector, los IDs de tags se resuelven con el catalogo incluido en
el servicio y se construye un documento ponderado: `tags x6`, `category x4`,
`description x3`, `name x1`. Direccion, ubicacion, `source` e IDs desconocidos no
participan en el embedding. `metadata.tag_ids` conserva trazabilidad de los IDs
originales y `metadata.tags` contiene sus nombres semanticos.

## Columnas

| Columna | Tipo | Descripcion |
|---|---|---|
| `external_id` | `TEXT` | ID del lugar en la API principal. |
| `document` | `TEXT` | Texto construido para generar el embedding. |
| `metadata` | `JSONB` | Datos estructurados utiles para filtros y respuesta. |
| `embedding` | `VECTOR(300)` | Promedio normalizado de embeddings FastText del `document`. |
| `content_hash` | `TEXT` | SHA-256 del contenido y la configuracion/version del embedding. |
| `embedding_model` | `TEXT` | Nombre del modelo usado para generar embeddings. |
| `embedding_version` | `TEXT` | Version logica del embedding. |
| `is_active` | `BOOLEAN` | Estado derivado desde la API principal. |
| `updated_at` | `TIMESTAMPTZ` | Fecha de ultima sincronizacion. |

## Metadata JSONB

El job guarda estos campos en `metadata` cuando existen:

```json
{
  "name": "Nombre del lugar",
  "category": "cafe",
  "category_label": "Cafeteria",
  "city": "Tuxtla Gutierrez",
  "state": "Chiapas",
  "source": "osm",
  "price_range": "$$",
  "is_active": true,
  "occasion": "pareja,amigos",
  "tags": "cafe,tranquilo",
  "short_description": "Descripcion corta...",
  "attribute_states": {
    "has_restrooms": true,
    "has_parking": false,
    "good_for_cooling_off": "yes"
  },
  "attribute_terms": ["banos y sanitarios", "refrescarse y nadar"],
  "negative_attribute_terms": ["estacionamiento"],
  "entertainment_features": ["Musica en vivo live music"],
  "contained_items": ["Alberca semiolimpica actividad acuatica"],
  "menu_items": ["Agua fresca bebidas"],
  "source_fields_present": ["name", "category", "facets"]
}
```

Los campos ausentes significan **desconocido**, no `false`. Por eso la falta de
descripcion, tags o atributos no resta confianza ni excluye automaticamente un lugar.
Solo `false` o `no` explicitos se guardan como evidencia negativa.

## Document

El campo `document` se construye concatenando, cuando existan:

```text
name category city state source price_range tags occasion description address
```

Ejemplo:

```text
Cafe Centro cafe Tuxtla Gutierrez Chiapas osm cafe tranquilo Un lugar para platicar Avenida Central
```

## Content Hash

`content_hash` se calcula con SHA-256 sobre el hash del contenido mas:

```json
{
  "source_content_hash": "...",
  "embedding_model": "facebook/fasttext-es-vectors",
  "embedding_version": "common-crawl-300-v1",
  "embedding_dimension": 300
}
```

Si el hash no cambia, el job omite regenerar embedding.

## Funciones Requeridas

La API NLP usa:

```sql
match_places(query_embedding VECTOR(300), match_count INTEGER, filters JSONB)
```

`filters.place_ids` acepta un arreglo de IDs generado por la consulta geografica a la API principal. `match_places` limita los resultados a esos `external_id`; no almacena coordenadas en esta tabla.

Los jobs usan:

```sql
upsert_place_embedding(
    p_external_id TEXT,
    p_document TEXT,
    p_metadata JSONB,
    p_embedding VECTOR(300),
    p_content_hash TEXT,
    p_embedding_model TEXT,
    p_embedding_version TEXT,
    p_is_active BOOLEAN
)
```

El SQL completo de referencia esta en:

```text
sql/aws_pgvector_contract.sql
```

## Facetas e incremental

Ejecuta despues de la migracion semantica:

```text
sql/migrations/20260721_03_place_facets_and_incremental_sync.sql
```

Esta migracion agrega indices de expresion sobre las facetas JSONB, filtros SQL
`required_attribute_states` y `*_any`, `place_sync_checkpoints` y la desactivacion
incremental. Es aditiva: no contiene `DROP`, `TRUNCATE` ni borrado de embeddings.
