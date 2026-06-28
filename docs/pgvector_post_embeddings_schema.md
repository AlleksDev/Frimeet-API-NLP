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
    embedding VECTOR(300) NOT NULL,
    content_hash TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

`VECTOR(300)` corresponde al modelo preentrenado
`facebook/fasttext-es-vectors` usado por `FastTextEmbeddingProvider`.

## Columnas

| Columna | Tipo | Descripcion |
|---|---|---|
| `external_id` | `TEXT` | ID del post en la API principal. |
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
match_posts(query_embedding VECTOR(300), match_count INTEGER, filters JSONB)
```

Los jobs usan:

```sql
upsert_post_embedding(
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
