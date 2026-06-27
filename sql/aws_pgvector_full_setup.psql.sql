-- Full setup for Frimeet API NLP pgvector storage.
-- Run with psql using an admin/RDS master role:
-- psql "host=<host> port=5432 dbname=postgres user=<admin> sslmode=require" -f sql/aws_pgvector_full_setup.psql.sql
--
-- Replace the passwords before running.
-- VECTOR(16) must match EMBEDDING_DIMENSION=16 in the API environment.

\set ON_ERROR_STOP on

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_owner') THEN
        CREATE ROLE nlp_owner LOGIN PASSWORD 'CAMBIA_OWNER_PASSWORD';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_reader') THEN
        CREATE ROLE nlp_reader LOGIN PASSWORD 'CAMBIA_READER_PASSWORD';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_writer') THEN
        CREATE ROLE nlp_writer LOGIN PASSWORD 'CAMBIA_WRITER_PASSWORD';
    END IF;
END $$;

SELECT 'CREATE DATABASE nlp_vectors OWNER nlp_owner'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'nlp_vectors')
\gexec

GRANT CONNECT ON DATABASE nlp_vectors TO nlp_reader, nlp_writer;
ALTER ROLE nlp_reader SET search_path = public;
ALTER ROLE nlp_writer SET search_path = public;

\connect nlp_vectors

CREATE EXTENSION IF NOT EXISTS vector;

REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO nlp_owner, nlp_reader, nlp_writer;

CREATE TABLE IF NOT EXISTS public.place_embeddings (
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

CREATE TABLE IF NOT EXISTS public.post_embeddings (
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
ON public.place_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS post_embeddings_embedding_hnsw_idx
ON public.post_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS place_embeddings_metadata_gin_idx
ON public.place_embeddings USING gin (metadata);

CREATE INDEX IF NOT EXISTS post_embeddings_metadata_gin_idx
ON public.post_embeddings USING gin (metadata);

CREATE OR REPLACE FUNCTION public.match_places(
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
    ORDER BY p.embedding <=> query_embedding
    LIMIT match_count;
$$;

CREATE OR REPLACE FUNCTION public.match_posts(
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

CREATE OR REPLACE FUNCTION public.upsert_place_embedding(
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

CREATE OR REPLACE FUNCTION public.upsert_post_embedding(
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

CREATE OR REPLACE FUNCTION public.get_place_content_hashes(p_external_ids TEXT[])
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

CREATE OR REPLACE FUNCTION public.get_post_content_hashes(p_external_ids TEXT[])
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

ALTER TABLE public.place_embeddings OWNER TO nlp_owner;
ALTER TABLE public.post_embeddings OWNER TO nlp_owner;

ALTER FUNCTION public.match_places(vector, integer, jsonb) OWNER TO nlp_owner;
ALTER FUNCTION public.match_posts(vector, integer, jsonb) OWNER TO nlp_owner;
ALTER FUNCTION public.upsert_place_embedding(text, text, jsonb, vector, text, text, text, boolean) OWNER TO nlp_owner;
ALTER FUNCTION public.upsert_post_embedding(text, text, jsonb, vector, text, text, text, boolean) OWNER TO nlp_owner;
ALTER FUNCTION public.get_place_content_hashes(text[]) OWNER TO nlp_owner;
ALTER FUNCTION public.get_post_content_hashes(text[]) OWNER TO nlp_owner;

REVOKE ALL ON public.place_embeddings FROM PUBLIC;
REVOKE ALL ON public.post_embeddings FROM PUBLIC;
REVOKE ALL ON FUNCTION public.match_places(vector, integer, jsonb) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.match_posts(vector, integer, jsonb) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.upsert_place_embedding(text, text, jsonb, vector, text, text, text, boolean) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.upsert_post_embedding(text, text, jsonb, vector, text, text, text, boolean) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_place_content_hashes(text[]) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_post_content_hashes(text[]) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION public.match_places(vector, integer, jsonb) TO nlp_reader;
GRANT EXECUTE ON FUNCTION public.match_posts(vector, integer, jsonb) TO nlp_reader;

GRANT EXECUTE ON FUNCTION public.upsert_place_embedding(text, text, jsonb, vector, text, text, text, boolean) TO nlp_writer;
GRANT EXECUTE ON FUNCTION public.upsert_post_embedding(text, text, jsonb, vector, text, text, text, boolean) TO nlp_writer;
GRANT EXECUTE ON FUNCTION public.get_place_content_hashes(text[]) TO nlp_writer;
GRANT EXECUTE ON FUNCTION public.get_post_content_hashes(text[]) TO nlp_writer;

SELECT to_regprocedure('public.match_places(vector, integer, jsonb)') AS match_places_signature;
SELECT to_regprocedure('public.match_posts(vector, integer, jsonb)') AS match_posts_signature;
SELECT has_function_privilege('nlp_reader', 'public.match_places(vector, integer, jsonb)', 'EXECUTE') AS reader_can_match_places;
SELECT has_function_privilege('nlp_reader', 'public.match_posts(vector, integer, jsonb)', 'EXECUTE') AS reader_can_match_posts;
