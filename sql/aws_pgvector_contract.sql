-- Reference contract for AWS RDS/Aurora PostgreSQL + pgvector.
-- Run this once with an admin/DBA role, not from Hugging Face.

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

CREATE TABLE IF NOT EXISTS user_embeddings (
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

CREATE TABLE IF NOT EXISTS club_embeddings (LIKE user_embeddings INCLUDING ALL);
CREATE TABLE IF NOT EXISTS group_embeddings (LIKE user_embeddings INCLUDING ALL);
CREATE TABLE IF NOT EXISTS event_embeddings (LIKE user_embeddings INCLUDING ALL);

ALTER TABLE place_embeddings ADD COLUMN IF NOT EXISTS textsearch TSVECTOR
GENERATED ALWAYS AS (to_tsvector('simple', document)) STORED;
ALTER TABLE post_embeddings ADD COLUMN IF NOT EXISTS textsearch TSVECTOR
GENERATED ALWAYS AS (to_tsvector('simple', document)) STORED;
ALTER TABLE user_embeddings ADD COLUMN IF NOT EXISTS textsearch TSVECTOR
GENERATED ALWAYS AS (to_tsvector('simple', document)) STORED;
ALTER TABLE club_embeddings ADD COLUMN IF NOT EXISTS textsearch TSVECTOR
GENERATED ALWAYS AS (to_tsvector('simple', document)) STORED;
ALTER TABLE group_embeddings ADD COLUMN IF NOT EXISTS textsearch TSVECTOR
GENERATED ALWAYS AS (to_tsvector('simple', document)) STORED;
ALTER TABLE event_embeddings ADD COLUMN IF NOT EXISTS textsearch TSVECTOR
GENERATED ALWAYS AS (to_tsvector('simple', document)) STORED;

CREATE INDEX IF NOT EXISTS place_embeddings_embedding_hnsw_idx
ON place_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS post_embeddings_embedding_hnsw_idx
ON post_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS place_embeddings_metadata_gin_idx
ON place_embeddings USING gin (metadata);

CREATE INDEX IF NOT EXISTS post_embeddings_metadata_gin_idx
ON post_embeddings USING gin (metadata);

CREATE INDEX IF NOT EXISTS user_embeddings_embedding_hnsw_idx
ON user_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS club_embeddings_embedding_hnsw_idx
ON club_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS group_embeddings_embedding_hnsw_idx
ON group_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS event_embeddings_embedding_hnsw_idx
ON event_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS user_embeddings_metadata_gin_idx ON user_embeddings USING gin (metadata);
CREATE INDEX IF NOT EXISTS club_embeddings_metadata_gin_idx ON club_embeddings USING gin (metadata);
CREATE INDEX IF NOT EXISTS group_embeddings_metadata_gin_idx ON group_embeddings USING gin (metadata);
CREATE INDEX IF NOT EXISTS event_embeddings_metadata_gin_idx ON event_embeddings USING gin (metadata);

CREATE INDEX IF NOT EXISTS place_embeddings_textsearch_gin_idx ON place_embeddings USING gin (textsearch);
CREATE INDEX IF NOT EXISTS post_embeddings_textsearch_gin_idx ON post_embeddings USING gin (textsearch);
CREATE INDEX IF NOT EXISTS user_embeddings_textsearch_gin_idx ON user_embeddings USING gin (textsearch);
CREATE INDEX IF NOT EXISTS club_embeddings_textsearch_gin_idx ON club_embeddings USING gin (textsearch);
CREATE INDEX IF NOT EXISTS group_embeddings_textsearch_gin_idx ON group_embeddings USING gin (textsearch);
CREATE INDEX IF NOT EXISTS event_embeddings_textsearch_gin_idx ON event_embeddings USING gin (textsearch);

CREATE OR REPLACE FUNCTION match_places(
    query_embedding VECTOR(300),
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
    ORDER BY p.embedding <=> query_embedding, p.external_id ASC
    LIMIT match_count;
$$;

CREATE OR REPLACE FUNCTION match_posts(
    query_embedding VECTOR(300),
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
    p_embedding VECTOR(300),
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
    p_embedding VECTOR(300),
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

CREATE OR REPLACE FUNCTION search_resource_embeddings(
    p_resource_type TEXT,
    p_query_text TEXT,
    p_query_embedding VECTOR(300),
    p_match_count INTEGER,
    p_filters JSONB DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    external_id TEXT,
    document TEXT,
    metadata JSONB,
    score DOUBLE PRECISION,
    semantic_score DOUBLE PRECISION,
    lexical_score DOUBLE PRECISION
)
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    target_table TEXT;
BEGIN
    target_table := CASE p_resource_type
        WHEN 'places' THEN 'place_embeddings'
        WHEN 'posts' THEN 'post_embeddings'
        WHEN 'users' THEN 'user_embeddings'
        WHEN 'clubs' THEN 'club_embeddings'
        WHEN 'groups' THEN 'group_embeddings'
        WHEN 'events' THEN 'event_embeddings'
        ELSE NULL
    END;
    IF target_table IS NULL THEN
        RAISE EXCEPTION 'Unsupported search resource type: %', p_resource_type;
    END IF;

    RETURN QUERY EXECUTE format($query$
        WITH eligible AS NOT MATERIALIZED (
            SELECT e.*
            FROM public.%I e
            WHERE e.is_active = true
              AND COALESCE(($4->>'is_active')::boolean, true) = true
              AND (
                    (
                        COALESCE((e.metadata->>'is_private')::boolean, false) = false
                        AND COALESCE((e.metadata->>'is_public')::boolean, true) = true
                    )
                    OR (
                        $4 ? 'requester_id'
                        AND (
                            e.metadata->>'creator_id' = $4->>'requester_id'
                            OR COALESCE(e.metadata->'authorized_user_ids', '[]'::jsonb)
                               ? ($4->>'requester_id')
                        )
                    )
              )
        ),
        vector_results AS (
            SELECT
                e.external_id,
                1 - (e.embedding <=> $1) AS semantic_score,
                row_number() OVER (
                    ORDER BY e.embedding <=> $1, e.external_id ASC
                ) AS vector_rank
            FROM eligible e
            ORDER BY e.embedding <=> $1, e.external_id ASC
            LIMIT GREATEST($3 * 3, 30)
        ),
        lexical_results AS (
            SELECT
                e.external_id,
                ts_rank_cd(e.textsearch, query.value) AS lexical_score,
                row_number() OVER (
                    ORDER BY ts_rank_cd(e.textsearch, query.value) DESC, e.external_id ASC
                ) AS lexical_rank
            FROM eligible e
            CROSS JOIN websearch_to_tsquery('simple', $2) AS query(value)
            WHERE e.textsearch @@ query.value
            ORDER BY lexical_score DESC, e.external_id ASC
            LIMIT GREATEST($3 * 3, 30)
        ),
        fused AS (
            SELECT
                COALESCE(v.external_id, l.external_id) AS external_id,
                v.semantic_score,
                l.lexical_score,
                COALESCE(1.0 / (60.0 + v.vector_rank), 0.0)
                    + COALESCE(1.0 / (60.0 + l.lexical_rank), 0.0) AS rrf_score
            FROM vector_results v
            FULL OUTER JOIN lexical_results l USING (external_id)
        )
        SELECT
            e.external_id,
            e.document,
            e.metadata,
            LEAST(
                1.0,
                f.rrf_score / (2.0 / 61.0)
                + CASE
                    WHEN lower(COALESCE(
                        e.metadata->>'search_title',
                        e.metadata->>'name',
                        e.metadata->>'title',
                        e.metadata->>'username',
                        ''
                    )) = lower($2) THEN 0.15
                    ELSE 0.0
                  END
            )::double precision AS score,
            f.semantic_score::double precision,
            f.lexical_score::double precision
        FROM fused f
        JOIN eligible e USING (external_id)
        ORDER BY score DESC, f.semantic_score DESC NULLS LAST, e.external_id ASC
        LIMIT GREATEST($3, 1)
    $query$, target_table)
    USING p_query_embedding, p_query_text, p_match_count, p_filters;
END;
$$;

CREATE OR REPLACE FUNCTION get_resource_content_hashes(
    p_resource_type TEXT,
    p_external_ids TEXT[]
)
RETURNS TABLE (external_id TEXT, content_hash TEXT)
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    target_table TEXT;
BEGIN
    target_table := CASE p_resource_type
        WHEN 'users' THEN 'user_embeddings'
        WHEN 'clubs' THEN 'club_embeddings'
        WHEN 'groups' THEN 'group_embeddings'
        WHEN 'events' THEN 'event_embeddings'
        ELSE NULL
    END;
    IF target_table IS NULL THEN
        RAISE EXCEPTION 'Unsupported sync resource type: %', p_resource_type;
    END IF;
    RETURN QUERY EXECUTE format(
        'SELECT e.external_id, e.content_hash FROM public.%I e WHERE e.external_id = ANY($1)',
        target_table
    ) USING p_external_ids;
END;
$$;

CREATE OR REPLACE FUNCTION upsert_resource_embedding(
    p_resource_type TEXT,
    p_external_id TEXT,
    p_document TEXT,
    p_metadata JSONB,
    p_embedding VECTOR(300),
    p_content_hash TEXT,
    p_embedding_model TEXT,
    p_embedding_version TEXT,
    p_is_active BOOLEAN
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    target_table TEXT;
BEGIN
    target_table := CASE p_resource_type
        WHEN 'users' THEN 'user_embeddings'
        WHEN 'clubs' THEN 'club_embeddings'
        WHEN 'groups' THEN 'group_embeddings'
        WHEN 'events' THEN 'event_embeddings'
        ELSE NULL
    END;
    IF target_table IS NULL THEN
        RAISE EXCEPTION 'Unsupported sync resource type: %', p_resource_type;
    END IF;
    EXECUTE format($query$
        INSERT INTO public.%I (
            external_id, document, metadata, embedding, content_hash,
            embedding_model, embedding_version, is_active, updated_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, now())
        ON CONFLICT (external_id) DO UPDATE SET
            document = EXCLUDED.document,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding,
            content_hash = EXCLUDED.content_hash,
            embedding_model = EXCLUDED.embedding_model,
            embedding_version = EXCLUDED.embedding_version,
            is_active = EXCLUDED.is_active,
            updated_at = now()
    $query$, target_table)
    USING p_external_id, p_document, p_metadata, p_embedding, p_content_hash,
          p_embedding_model, p_embedding_version, p_is_active;
END;
$$;

-- Example least-privilege grants. Adjust schema/database/user creation to your setup.
-- GRANT EXECUTE ON FUNCTION match_places(VECTOR, INTEGER, JSONB) TO nlp_reader;
-- GRANT EXECUTE ON FUNCTION match_posts(VECTOR, INTEGER, JSONB) TO nlp_reader;
-- GRANT EXECUTE ON FUNCTION get_place_content_hashes(TEXT[]) TO nlp_writer;
-- GRANT EXECUTE ON FUNCTION get_post_content_hashes(TEXT[]) TO nlp_writer;
-- GRANT EXECUTE ON FUNCTION upsert_place_embedding(TEXT, TEXT, JSONB, VECTOR, TEXT, TEXT, TEXT, BOOLEAN) TO nlp_writer;
-- GRANT EXECUTE ON FUNCTION upsert_post_embedding(TEXT, TEXT, JSONB, VECTOR, TEXT, TEXT, TEXT, BOOLEAN) TO nlp_writer;
-- GRANT EXECUTE ON FUNCTION search_resource_embeddings(TEXT, TEXT, VECTOR, INTEGER, JSONB) TO nlp_reader;
-- GRANT EXECUTE ON FUNCTION get_resource_content_hashes(TEXT, TEXT[]) TO nlp_writer;
-- GRANT EXECUTE ON FUNCTION upsert_resource_embedding(TEXT, TEXT, TEXT, JSONB, VECTOR, TEXT, TEXT, TEXT, BOOLEAN) TO nlp_writer;
