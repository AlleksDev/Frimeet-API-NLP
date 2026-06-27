-- Reference contract for AWS RDS/Aurora PostgreSQL + pgvector.
-- Run this once with an admin/DBA role, not from Hugging Face.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS place_embeddings (
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

CREATE INDEX IF NOT EXISTS place_embeddings_embedding_hnsw_idx
ON place_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS post_embeddings_embedding_hnsw_idx
ON post_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS place_embeddings_metadata_gin_idx
ON place_embeddings USING gin (metadata);

CREATE INDEX IF NOT EXISTS post_embeddings_metadata_gin_idx
ON post_embeddings USING gin (metadata);

CREATE OR REPLACE FUNCTION match_places(
    query_embedding VECTOR(16),
    match_count INTEGER,
    filters JSONB DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    external_id TEXT,
    document TEXT,
    metadata JSONB,
    score DOUBLE PRECISION
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT
        p.external_id,
        p.document,
        p.metadata,
        1 - (p.embedding <=> query_embedding) AS score
    FROM public.place_embeddings p
    WHERE p.is_active = true
      AND COALESCE((filters->>'is_active')::boolean, true) = true
      AND ((filters ? 'city') IS FALSE OR lower(p.metadata->>'city') = lower(filters->>'city'))
      AND ((filters ? 'state') IS FALSE OR lower(p.metadata->>'state') = lower(filters->>'state'))
      AND ((filters ? 'category') IS FALSE OR lower(p.metadata->>'category') = lower(filters->>'category'))
      AND ((filters ? 'price_range') IS FALSE OR p.metadata->>'price_range' = filters->>'price_range')
      AND ((filters ? 'occasion') IS FALSE OR p.metadata->>'occasion' ILIKE ('%' || (filters->>'occasion') || '%'))
      AND (
          (filters ? 'place_ids') IS FALSE
          OR p.external_id IN (SELECT jsonb_array_elements_text(filters->'place_ids'))
      )
    ORDER BY p.embedding <=> query_embedding
    LIMIT match_count;
$$;

CREATE OR REPLACE FUNCTION match_posts(
    query_embedding VECTOR(16),
    match_count INTEGER,
    filters JSONB DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    external_id TEXT,
    document TEXT,
    metadata JSONB,
    score DOUBLE PRECISION
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT
        p.external_id,
        p.document,
        p.metadata,
        1 - (p.embedding <=> query_embedding) AS score
    FROM public.post_embeddings p
    WHERE p.is_active = true
      AND COALESCE((filters->>'is_active')::boolean, true) = true
      AND ((filters ? 'city') IS FALSE OR lower(p.metadata->>'city') = lower(filters->>'city'))
    ORDER BY p.embedding <=> query_embedding
    LIMIT match_count;
$$;

CREATE OR REPLACE FUNCTION upsert_place_embedding(
    p_external_id TEXT,
    p_document TEXT,
    p_metadata JSONB,
    p_embedding VECTOR(16),
    p_content_hash TEXT,
    p_embedding_model TEXT,
    p_embedding_version TEXT,
    p_is_active BOOLEAN
)
RETURNS VOID
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    INSERT INTO public.place_embeddings (
        external_id,
        document,
        metadata,
        embedding,
        content_hash,
        embedding_model,
        embedding_version,
        is_active,
        updated_at
    )
    VALUES (
        p_external_id,
        p_document,
        p_metadata,
        p_embedding,
        p_content_hash,
        p_embedding_model,
        p_embedding_version,
        p_is_active,
        now()
    )
    ON CONFLICT (external_id) DO UPDATE SET
        document = EXCLUDED.document,
        metadata = EXCLUDED.metadata,
        embedding = EXCLUDED.embedding,
        content_hash = EXCLUDED.content_hash,
        embedding_model = EXCLUDED.embedding_model,
        embedding_version = EXCLUDED.embedding_version,
        is_active = EXCLUDED.is_active,
        updated_at = now();
$$;

CREATE OR REPLACE FUNCTION upsert_post_embedding(
    p_external_id TEXT,
    p_document TEXT,
    p_metadata JSONB,
    p_embedding VECTOR(16),
    p_content_hash TEXT,
    p_embedding_model TEXT,
    p_embedding_version TEXT,
    p_is_active BOOLEAN
)
RETURNS VOID
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    INSERT INTO public.post_embeddings (
        external_id,
        document,
        metadata,
        embedding,
        content_hash,
        embedding_model,
        embedding_version,
        is_active,
        updated_at
    )
    VALUES (
        p_external_id,
        p_document,
        p_metadata,
        p_embedding,
        p_content_hash,
        p_embedding_model,
        p_embedding_version,
        p_is_active,
        now()
    )
    ON CONFLICT (external_id) DO UPDATE SET
        document = EXCLUDED.document,
        metadata = EXCLUDED.metadata,
        embedding = EXCLUDED.embedding,
        content_hash = EXCLUDED.content_hash,
        embedding_model = EXCLUDED.embedding_model,
        embedding_version = EXCLUDED.embedding_version,
        is_active = EXCLUDED.is_active,
        updated_at = now();
$$;

CREATE OR REPLACE FUNCTION get_place_content_hashes(p_external_ids TEXT[])
RETURNS TABLE (
    external_id TEXT,
    content_hash TEXT
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT p.external_id, p.content_hash
    FROM public.place_embeddings p
    WHERE p.external_id = ANY(p_external_ids);
$$;

CREATE OR REPLACE FUNCTION get_post_content_hashes(p_external_ids TEXT[])
RETURNS TABLE (
    external_id TEXT,
    content_hash TEXT
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT p.external_id, p.content_hash
    FROM public.post_embeddings p
    WHERE p.external_id = ANY(p_external_ids);
$$;

-- Example least-privilege grants. Adjust schema/database/user creation to your setup.
-- GRANT EXECUTE ON FUNCTION match_places(VECTOR, INTEGER, JSONB) TO nlp_reader;
-- GRANT EXECUTE ON FUNCTION match_posts(VECTOR, INTEGER, JSONB) TO nlp_reader;
-- GRANT EXECUTE ON FUNCTION get_place_content_hashes(TEXT[]) TO nlp_writer;
-- GRANT EXECUTE ON FUNCTION get_post_content_hashes(TEXT[]) TO nlp_writer;
-- GRANT EXECUTE ON FUNCTION upsert_place_embedding(TEXT, TEXT, JSONB, VECTOR, TEXT, TEXT, TEXT, BOOLEAN) TO nlp_writer;
-- GRANT EXECUTE ON FUNCTION upsert_post_embedding(TEXT, TEXT, JSONB, VECTOR, TEXT, TEXT, TEXT, BOOLEAN) TO nlp_writer;
