# PgVector Post Embeddings Schema

Este es el contrato actual para guardar publicaciones derivadas en RDS PostgreSQL + pgvector.

La API principal sigue siendo la fuente de verdad. Esta tabla contiene datos derivados para busqueda semantica y recomendacion de posts.

## Tabla

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS post_embeddings (
    external_id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(16) NOT NULL,
    content_hash TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

`VECTOR(16)` corresponde al `MockEmbeddingProvider` actual.

Cuando se cambie a embeddings reales, hay que cambiar:

```env
EMBEDDING_DIMENSION=<dimension_real>
```

y tambien:

```sql
embedding VECTOR(<dimension_real>)
```

## Columnas

| Columna | Tipo | Descripcion |
|---|---|---|
| `external_id` | `TEXT` | ID del post en la API principal. |
| `document` | `TEXT` | Texto construido para generar el embedding. |
| `metadata` | `JSONB` | Datos estructurados utiles para filtros y respuesta. |
| `embedding` | `VECTOR(16)` | Embedding del `document`. |
| `content_hash` | `TEXT` | SHA-256 estable de document + metadata + is_active. |
| `embedding_model` | `TEXT` | Nombre del modelo usado para generar embeddings. |
| `embedding_version` | `TEXT` | Version logica del embedding. |
| `is_active` | `BOOLEAN` | Estado derivado desde la API principal. |
| `updated_at` | `TIMESTAMPTZ` | Fecha de ultima sincronizacion. |

## Metadata JSONB

El job guarda estos campos en `metadata` cuando existen:

```json
{
  "title": "Titulo del post",
  "city": "Tuxtla Gutierrez",
  "state": "Chiapas",
  "source": "internal",
  "tags": "cafe,amigos",
  "is_active": true
}
```

## Document

El campo `document` se construye concatenando, cuando existan:

```text
title city state source tags text/content/description/body
```

Ejemplo:

```text
Plan de cafe Tuxtla Gutierrez Chiapas internal cafe amigos Una publicacion para salir por cafe
```

## Content Hash

`content_hash` se calcula con SHA-256 sobre:

```json
{
  "document": "...",
  "metadata": {...},
  "is_active": true
}
```

Si el hash no cambia, el job omite regenerar embedding.

## Funciones Requeridas

La API NLP usa:

```sql
match_posts(query_embedding VECTOR(16), match_count INTEGER, filters JSONB)
```

Los jobs usan:

```sql
upsert_post_embedding(
    p_external_id TEXT,
    p_document TEXT,
    p_metadata JSONB,
    p_embedding VECTOR(16),
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
