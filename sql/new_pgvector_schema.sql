-- Esquema convergente completo de API-NLP para pgAdmin/DataGrid.
-- Sirve para una BD vacia o una BD existente que ya use VECTOR(300).
-- NO convierte VECTOR(16): usa la migracion FastText separada para ese caso.

DO $bootstrap$
DECLARE actual_type TEXT;
BEGIN
    IF to_regclass('public.post_embeddings') IS NOT NULL THEN
        SELECT format_type(attribute.atttypid, attribute.atttypmod)
        INTO actual_type
        FROM pg_attribute attribute
        WHERE attribute.attrelid = 'public.post_embeddings'::regclass
          AND attribute.attname = 'embedding' AND NOT attribute.attisdropped;
        IF actual_type IS DISTINCT FROM 'vector(300)' THEN
            RAISE EXCEPTION 'Se requiere vector(300), actual=%', actual_type;
        END IF;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_owner') THEN
        EXECUTE 'CREATE ROLE nlp_owner NOLOGIN';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_reader') THEN
        EXECUTE 'CREATE ROLE nlp_reader LOGIN';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_writer') THEN
        EXECUTE 'CREATE ROLE nlp_writer LOGIN';
    END IF;
END
$bootstrap$;

SET search_path = public;

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

-- Post-feed derived storage. The versioned companion migration defines the
-- controlled functions, revokes and grants used by the adapters.
CREATE TABLE IF NOT EXISTS post_cluster_runs (
    id UUID PRIMARY KEY,
    algorithm TEXT NOT NULL CHECK (algorithm = 'kmeans'),
    algorithm_version TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension = 300),
    k INTEGER NOT NULL CHECK (k > 1),
    sample_size INTEGER NOT NULL CHECK (sample_size >= k),
    inertia DOUBLE PRECISION,
    silhouette_score DOUBLE PRECISION,
    status TEXT NOT NULL CHECK (status IN ('running', 'validated', 'active', 'failed', 'superseded')),
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    activated_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS post_cluster_runs_one_active_idx
ON post_cluster_runs ((status)) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS post_clusters (
    run_id UUID NOT NULL REFERENCES post_cluster_runs(id) ON DELETE CASCADE,
    cluster_id INTEGER NOT NULL,
    centroid VECTOR(300) NOT NULL,
    label TEXT,
    size INTEGER NOT NULL CHECK (size >= 0),
    representative_post_ids TEXT[] NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (run_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS post_cluster_memberships (
    run_id UUID NOT NULL,
    post_id TEXT NOT NULL REFERENCES post_embeddings(external_id) ON DELETE CASCADE,
    cluster_id INTEGER NOT NULL,
    distance_to_centroid DOUBLE PRECISION NOT NULL CHECK (distance_to_centroid >= 0),
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, post_id),
    FOREIGN KEY (run_id, cluster_id) REFERENCES post_clusters(run_id, cluster_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_interest_embeddings (
    user_id TEXT PRIMARY KEY,
    embedding VECTOR(300) NOT NULL,
    weighted_sum VECTOR(300) NOT NULL,
    total_weight DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (total_weight >= 0),
    source_hash TEXT NOT NULL,
    profile_version TEXT NOT NULL,
    interaction_count INTEGER NOT NULL DEFAULT 0 CHECK (interaction_count >= 0),
    last_event_id BIGINT NOT NULL DEFAULT 0 CHECK (last_event_id >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS post_sync_checkpoints (
    consumer TEXT PRIMARY KEY,
    last_event_id BIGINT NOT NULL DEFAULT 0 CHECK (last_event_id >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS place_sync_checkpoints (
    consumer TEXT PRIMARY KEY,
    last_event_id BIGINT NOT NULL DEFAULT 0 CHECK (last_event_id >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS post_embedding_tombstones (
    post_id TEXT PRIMARY KEY,
    source_version BIGINT NOT NULL CHECK (source_version >= 0),
    deleted_at TIMESTAMPTZ NOT NULL DEFAULT now()
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
    author_type TEXT,
    author_id TEXT,
    published_at TIMESTAMPTZ,
    source_version BIGINT,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
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
      AND (
          (filters ? 'categories') IS FALSE
          OR EXISTS (
              SELECT 1
              FROM jsonb_array_elements_text(filters->'categories') AS allowed(value)
              WHERE lower(p.metadata->>'category') = lower(allowed.value)
          )
      )
      AND ((filters ? 'price_range') IS FALSE OR p.metadata->>'price_range' = filters->>'price_range')
      AND (
          (filters ? 'price_ranges') IS FALSE
          OR EXISTS (
              SELECT 1
              FROM jsonb_array_elements_text(filters->'price_ranges') AS allowed(value)
              WHERE p.metadata->>'price_range' = allowed.value
          )
      )
      AND (
          (filters ? 'tags') IS FALSE
          OR EXISTS (
              SELECT 1
              FROM jsonb_array_elements_text(filters->'tags') AS allowed(value)
              WHERE lower(COALESCE(p.metadata->>'tags', ''))
                    LIKE ('%' || lower(allowed.value) || '%')
          )
      )
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

CREATE OR REPLACE FUNCTION deactivate_place_embedding(p_external_id TEXT)
RETURNS VOID
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    UPDATE public.place_embeddings
    SET is_active = FALSE,
        updated_at = now()
    WHERE external_id = p_external_id;
$$;

CREATE OR REPLACE FUNCTION upsert_post_embedding(
    p_external_id TEXT,
    p_document TEXT,
    p_metadata JSONB,
    p_embedding VECTOR(300),
    p_content_hash TEXT,
    p_embedding_model TEXT,
    p_embedding_version TEXT,
    p_is_active BOOLEAN,
    p_author_type TEXT,
    p_author_id TEXT,
    p_published_at TIMESTAMPTZ,
    p_source_version BIGINT
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM public.post_embedding_tombstones tombstone
        WHERE tombstone.post_id = p_external_id
          AND (p_source_version IS NULL OR tombstone.source_version >= p_source_version)
    ) THEN
        RETURN;
    END IF;
    IF p_source_version IS NOT NULL THEN
        DELETE FROM public.post_embedding_tombstones
        WHERE post_id = p_external_id AND source_version < p_source_version;
    END IF;
    INSERT INTO public.post_embeddings (
        external_id, document, metadata, embedding, content_hash,
        embedding_model, embedding_version, is_active, author_type,
        author_id, published_at, source_version, indexed_at, updated_at
    )
    VALUES (
        p_external_id, p_document, p_metadata, p_embedding, p_content_hash,
        p_embedding_model, p_embedding_version, p_is_active, p_author_type,
        p_author_id, p_published_at, p_source_version, now(), now()
    )
    ON CONFLICT (external_id) DO UPDATE SET
        document = EXCLUDED.document,
        metadata = EXCLUDED.metadata,
        embedding = EXCLUDED.embedding,
        content_hash = EXCLUDED.content_hash,
        embedding_model = EXCLUDED.embedding_model,
        embedding_version = EXCLUDED.embedding_version,
        is_active = EXCLUDED.is_active,
        author_type = EXCLUDED.author_type,
        author_id = EXCLUDED.author_id,
        published_at = EXCLUDED.published_at,
        source_version = EXCLUDED.source_version,
        indexed_at = now(),
        updated_at = now()
    WHERE EXCLUDED.source_version IS NULL
       OR post_embeddings.source_version IS NULL
       OR EXCLUDED.source_version >= post_embeddings.source_version;
END;
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

CREATE OR REPLACE FUNCTION try_parse_timestamptz(p_value TEXT)
RETURNS TIMESTAMPTZ
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
SET search_path = pg_catalog
AS $$
BEGIN
    RETURN NULLIF(btrim(p_value), '')::timestamptz;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION try_parse_positive_integer(p_value TEXT)
RETURNS INTEGER
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
SET search_path = pg_catalog
AS $$
DECLARE
    parsed INTEGER;
BEGIN
    parsed := NULLIF(btrim(p_value), '')::integer;
    RETURN CASE WHEN parsed > 0 THEN parsed ELSE NULL END;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
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
                    ($4 ? 'city') IS FALSE
                    OR lower(e.metadata->>'city') = lower($4->>'city')
              )
              AND (
                    ($4 ? 'state') IS FALSE
                    OR lower(e.metadata->>'state') = lower($4->>'state')
              )
              AND (
                    ($4 ? 'categories') IS FALSE
                    OR EXISTS (
                        SELECT 1
                        FROM jsonb_array_elements_text($4->'categories') AS allowed(value)
                        WHERE lower(e.metadata->>'category') = lower(allowed.value)
                    )
              )
              AND (
                    ($4 ? 'tags') IS FALSE
                    OR EXISTS (
                        SELECT 1
                        FROM jsonb_array_elements_text($4->'tags') AS allowed(value)
                        WHERE lower(COALESCE(e.metadata->>'tags', ''))
                              LIKE ('%%' || lower(allowed.value) || '%%')
                    )
              )
              AND (
                    ($4 ? 'published_from') IS FALSE
                    OR public.try_parse_timestamptz(e.metadata->>'published_at')
                       >= ($4->>'published_from')::timestamptz
              )
              AND (
                    ($4 ? 'published_to') IS FALSE
                    OR public.try_parse_timestamptz(e.metadata->>'published_at')
                       <= ($4->>'published_to')::timestamptz
              )
              AND (
                    ($4 ? 'event_from') IS FALSE
                    OR public.try_parse_timestamptz(e.metadata->>'start_time')
                       >= ($4->>'event_from')::timestamptz
              )
              AND (
                    ($4 ? 'event_to') IS FALSE
                    OR public.try_parse_timestamptz(e.metadata->>'start_time')
                       <= ($4->>'event_to')::timestamptz
              )
              AND (
                    ($4 ? 'event_active_at') IS FALSE
                    OR (
                        public.try_parse_timestamptz(e.metadata->>'start_time') IS NOT NULL
                        AND public.try_parse_positive_integer(
                            e.metadata->>'duration_minutes'
                        ) IS NOT NULL
                        AND public.try_parse_timestamptz(e.metadata->>'start_time')
                            + make_interval(
                                mins => public.try_parse_positive_integer(
                                    e.metadata->>'duration_minutes'
                                )
                              )
                            > ($4->>'event_active_at')::timestamptz
                    )
              )
              AND (
                    ($4 ? 'is_online') IS FALSE
                    OR COALESCE((e.metadata->>'is_online')::boolean, false)
                       = ($4->>'is_online')::boolean
              )
              AND (
                    ($4 ? 'user_roles') IS FALSE
                    OR EXISTS (
                        SELECT 1
                        FROM jsonb_array_elements_text($4->'user_roles') AS allowed(value)
                        WHERE lower(e.metadata->>'role') = lower(allowed.value)
                    )
              )
              AND (
                    ($4 ? 'place_ids') IS FALSE
                    OR e.metadata->>'place_id' IN (
                        SELECT jsonb_array_elements_text($4->'place_ids')
                    )
              )
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
        WHERE (
            (
                ($4 ? 'min_semantic_score') IS FALSE
                AND ($4 ? 'min_lexical_score') IS FALSE
            )
            OR (
                ($4 ? 'min_semantic_score')
                AND COALESCE(
                    f.semantic_score >= ($4->>'min_semantic_score')::double precision,
                    FALSE
                )
            )
            OR (
                ($4 ? 'min_lexical_score')
                AND COALESCE(
                    f.lexical_score >= ($4->>'min_lexical_score')::double precision,
                    FALSE
                )
            )
        )
        ORDER BY score DESC, f.semantic_score DESC NULLS LAST, e.external_id ASC
        LIMIT GREATEST($3, 1)
    $query$, target_table)
    USING p_query_embedding, p_query_text, p_match_count, p_filters;
END;
$$;

REVOKE ALL ON FUNCTION public.try_parse_timestamptz(text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.try_parse_positive_integer(text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.try_parse_timestamptz(text) TO nlp_reader;
GRANT EXECUTE ON FUNCTION public.try_parse_positive_integer(text) TO nlp_reader;

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
-- GRANT EXECUTE ON FUNCTION upsert_post_embedding(TEXT, TEXT, JSONB, VECTOR, TEXT, TEXT, TEXT, BOOLEAN, TEXT, TEXT, TIMESTAMPTZ, BIGINT) TO nlp_writer;
-- GRANT EXECUTE ON FUNCTION search_resource_embeddings(TEXT, TEXT, VECTOR, INTEGER, JSONB) TO nlp_reader;
-- GRANT EXECUTE ON FUNCTION get_resource_content_hashes(TEXT, TEXT[]) TO nlp_writer;
-- GRANT EXECUTE ON FUNCTION upsert_resource_embedding(TEXT, TEXT, TEXT, JSONB, VECTOR, TEXT, TEXT, TEXT, BOOLEAN) TO nlp_writer;

-- The feed-specific tables and controlled functions are versioned in
-- sql/migrate_post_feed_v1.sql. Apply that companion contract immediately
-- after this base contract. Keeping the feed extension versioned prevents a
-- fresh setup and an upgraded database from drifting apart; the full psql
-- setup includes it automatically with \ir.


BEGIN;

-- Derived post data. Source-of-truth permissions remain in the Go API.
ALTER TABLE public.post_embeddings
    ADD COLUMN IF NOT EXISTS author_type TEXT,
    ADD COLUMN IF NOT EXISTS author_id TEXT,
    ADD COLUMN IF NOT EXISTS published_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS source_version BIGINT,
    ADD COLUMN IF NOT EXISTS indexed_at TIMESTAMPTZ NOT NULL DEFAULT now();

CREATE INDEX IF NOT EXISTS post_embeddings_active_published_idx
ON public.post_embeddings (published_at DESC, external_id DESC)
WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS post_embeddings_author_idx
ON public.post_embeddings (author_type, author_id)
WHERE is_active = TRUE;

CREATE TABLE IF NOT EXISTS public.post_cluster_runs (
    id UUID PRIMARY KEY,
    algorithm TEXT NOT NULL CHECK (algorithm = 'kmeans'),
    algorithm_version TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension = 300),
    k INTEGER NOT NULL CHECK (k > 1),
    sample_size INTEGER NOT NULL CHECK (sample_size >= k),
    inertia DOUBLE PRECISION,
    silhouette_score DOUBLE PRECISION,
    status TEXT NOT NULL CHECK (status IN ('running', 'validated', 'active', 'failed', 'superseded')),
    parameters JSONB NOT NULL DEFAULT '{}'::jsonb,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    activated_at TIMESTAMPTZ,
    error_message TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS post_cluster_runs_one_active_idx
ON public.post_cluster_runs ((status)) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS public.post_clusters (
    run_id UUID NOT NULL REFERENCES public.post_cluster_runs(id) ON DELETE CASCADE,
    cluster_id INTEGER NOT NULL,
    centroid VECTOR(300) NOT NULL,
    label TEXT,
    size INTEGER NOT NULL CHECK (size >= 0),
    representative_post_ids TEXT[] NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (run_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS public.post_cluster_memberships (
    run_id UUID NOT NULL,
    post_id TEXT NOT NULL REFERENCES public.post_embeddings(external_id) ON DELETE CASCADE,
    cluster_id INTEGER NOT NULL,
    distance_to_centroid DOUBLE PRECISION NOT NULL CHECK (distance_to_centroid >= 0),
    assigned_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, post_id),
    FOREIGN KEY (run_id, cluster_id)
        REFERENCES public.post_clusters(run_id, cluster_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS post_cluster_memberships_cluster_idx
ON public.post_cluster_memberships (run_id, cluster_id, distance_to_centroid);

CREATE TABLE IF NOT EXISTS public.user_interest_embeddings (
    user_id TEXT PRIMARY KEY,
    embedding VECTOR(300) NOT NULL,
    weighted_sum VECTOR(300) NOT NULL,
    total_weight DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (total_weight >= 0),
    source_hash TEXT NOT NULL,
    profile_version TEXT NOT NULL,
    interaction_count INTEGER NOT NULL DEFAULT 0 CHECK (interaction_count >= 0),
    last_event_id BIGINT NOT NULL DEFAULT 0 CHECK (last_event_id >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE public.user_interest_embeddings
    ADD COLUMN IF NOT EXISTS weighted_sum VECTOR(300),
    ADD COLUMN IF NOT EXISTS total_weight DOUBLE PRECISION NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_event_id BIGINT NOT NULL DEFAULT 0;

UPDATE public.user_interest_embeddings
SET weighted_sum = embedding
WHERE weighted_sum IS NULL;

ALTER TABLE public.user_interest_embeddings
    ALTER COLUMN weighted_sum SET NOT NULL;

CREATE TABLE IF NOT EXISTS public.post_sync_checkpoints (
    consumer TEXT PRIMARY KEY,
    last_event_id BIGINT NOT NULL DEFAULT 0 CHECK (last_event_id >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.post_embedding_tombstones (
    post_id TEXT PRIMARY KEY,
    source_version BIGINT NOT NULL CHECK (source_version >= 0),
    deleted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Replace the legacy eight-argument function with the version-aware contract.
DROP FUNCTION IF EXISTS public.upsert_post_embedding(
    TEXT, TEXT, JSONB, VECTOR, TEXT, TEXT, TEXT, BOOLEAN
);

CREATE OR REPLACE FUNCTION public.upsert_post_embedding(
    p_external_id TEXT,
    p_document TEXT,
    p_metadata JSONB,
    p_embedding VECTOR(300),
    p_content_hash TEXT,
    p_embedding_model TEXT,
    p_embedding_version TEXT,
    p_is_active BOOLEAN,
    p_author_type TEXT,
    p_author_id TEXT,
    p_published_at TIMESTAMPTZ,
    p_source_version BIGINT
) RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM public.post_embedding_tombstones tombstone
        WHERE tombstone.post_id = p_external_id
          AND (p_source_version IS NULL OR tombstone.source_version >= p_source_version)
    ) THEN
        RETURN;
    END IF;

    IF p_source_version IS NOT NULL THEN
        DELETE FROM public.post_embedding_tombstones
        WHERE post_id = p_external_id
          AND source_version < p_source_version;
    END IF;

    INSERT INTO public.post_embeddings (
        external_id, document, metadata, embedding, content_hash,
        embedding_model, embedding_version, is_active, author_type,
        author_id, published_at, source_version, indexed_at, updated_at
    ) VALUES (
        p_external_id, p_document, p_metadata, p_embedding, p_content_hash,
        p_embedding_model, p_embedding_version, p_is_active, p_author_type,
        p_author_id, p_published_at, p_source_version, now(), now()
    )
    ON CONFLICT (external_id) DO UPDATE SET
        document = EXCLUDED.document,
        metadata = EXCLUDED.metadata,
        embedding = EXCLUDED.embedding,
        content_hash = EXCLUDED.content_hash,
        embedding_model = EXCLUDED.embedding_model,
        embedding_version = EXCLUDED.embedding_version,
        is_active = EXCLUDED.is_active,
        author_type = EXCLUDED.author_type,
        author_id = EXCLUDED.author_id,
        published_at = EXCLUDED.published_at,
        source_version = EXCLUDED.source_version,
        indexed_at = now(),
        updated_at = now()
    WHERE EXCLUDED.source_version IS NULL
       OR post_embeddings.source_version IS NULL
       OR EXCLUDED.source_version >= post_embeddings.source_version;
END;
$$;

CREATE OR REPLACE FUNCTION public.deactivate_post_embedding(
    p_external_id TEXT,
    p_source_version BIGINT
) RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO public.post_embedding_tombstones (post_id, source_version, deleted_at)
    VALUES (p_external_id, p_source_version, now())
    ON CONFLICT (post_id) DO UPDATE SET
        source_version = GREATEST(
            post_embedding_tombstones.source_version,
            EXCLUDED.source_version
        ),
        deleted_at = CASE
            WHEN EXCLUDED.source_version >= post_embedding_tombstones.source_version
            THEN now()
            ELSE post_embedding_tombstones.deleted_at
        END;

    UPDATE public.post_embeddings
    SET is_active = FALSE,
        source_version = GREATEST(COALESCE(source_version, 0), p_source_version),
        indexed_at = now(),
        updated_at = now()
    WHERE external_id = p_external_id
      AND (source_version IS NULL OR p_source_version >= source_version);
END;
$$;

CREATE OR REPLACE FUNCTION public.get_post_sync_checkpoint(p_consumer TEXT)
RETURNS BIGINT
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT COALESCE(
        (SELECT last_event_id FROM public.post_sync_checkpoints WHERE consumer = p_consumer),
        0
    );
$$;

CREATE OR REPLACE FUNCTION public.save_post_sync_checkpoint(
    p_consumer TEXT,
    p_last_event_id BIGINT
) RETURNS VOID
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    INSERT INTO public.post_sync_checkpoints (consumer, last_event_id, updated_at)
    VALUES (p_consumer, p_last_event_id, now())
    ON CONFLICT (consumer) DO UPDATE SET
        last_event_id = GREATEST(post_sync_checkpoints.last_event_id, EXCLUDED.last_event_id),
        updated_at = now();
$$;

CREATE OR REPLACE FUNCTION public.reset_post_sync_checkpoint(p_consumer TEXT)
RETURNS VOID
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    DELETE FROM public.post_sync_checkpoints WHERE consumer = p_consumer;
$$;

CREATE OR REPLACE FUNCTION public.get_place_sync_checkpoint(p_consumer TEXT)
RETURNS BIGINT
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT COALESCE(
        (SELECT last_event_id FROM public.place_sync_checkpoints WHERE consumer = p_consumer),
        0
    );
$$;

CREATE OR REPLACE FUNCTION public.save_place_sync_checkpoint(
    p_consumer TEXT,
    p_last_event_id BIGINT
) RETURNS VOID
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    INSERT INTO public.place_sync_checkpoints (consumer, last_event_id, updated_at)
    VALUES (p_consumer, p_last_event_id, now())
    ON CONFLICT (consumer) DO UPDATE SET
        last_event_id = GREATEST(place_sync_checkpoints.last_event_id, EXCLUDED.last_event_id),
        updated_at = now();
$$;

CREATE OR REPLACE FUNCTION public.get_post_feed_features(
    p_user_id TEXT,
    p_post_ids TEXT[]
) RETURNS TABLE (
    external_id TEXT,
    semantic_similarity DOUBLE PRECISION,
    cluster_id INTEGER,
    cluster_run_id TEXT,
    cold_start BOOLEAN,
    post_embedding TEXT
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    WITH requested AS (
        SELECT post_id, ordinality
        FROM unnest(p_post_ids) WITH ORDINALITY AS ids(post_id, ordinality)
    ), active_run AS (
        SELECT id FROM public.post_cluster_runs
        WHERE status = 'active'
        ORDER BY activated_at DESC NULLS LAST
        LIMIT 1
    ), profile AS (
        SELECT embedding FROM public.user_interest_embeddings WHERE user_id = p_user_id
    )
    SELECT requested.post_id,
           CASE WHEN profile.embedding IS NULL OR post.embedding IS NULL THEN 0.0
                ELSE 1 - (post.embedding <=> profile.embedding) END,
           membership.cluster_id,
           active_run.id::text,
           profile.embedding IS NULL,
           post.embedding::text
    FROM requested
    LEFT JOIN public.post_embeddings post
      ON post.external_id = requested.post_id AND post.is_active = TRUE
    LEFT JOIN profile ON TRUE
    LEFT JOIN active_run ON TRUE
    LEFT JOIN public.post_cluster_memberships membership
      ON membership.post_id = post.external_id AND membership.run_id = active_run.id
    ORDER BY requested.ordinality;
$$;

CREATE OR REPLACE FUNCTION public.get_post_embeddings_for_profile(p_post_ids TEXT[])
RETURNS TABLE (post_id TEXT, embedding VECTOR(300))
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT external_id, embedding
    FROM public.post_embeddings
    WHERE external_id = ANY(p_post_ids) AND is_active = TRUE;
$$;

CREATE OR REPLACE FUNCTION public.get_user_interest_states(p_user_ids TEXT[])
RETURNS TABLE (
    user_id TEXT,
    embedding VECTOR(300),
    weighted_sum VECTOR(300),
    total_weight DOUBLE PRECISION,
    interaction_count INTEGER,
    last_event_id BIGINT
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT profile.user_id, profile.embedding, profile.weighted_sum,
           profile.total_weight, profile.interaction_count, profile.last_event_id
    FROM public.user_interest_embeddings profile
    WHERE profile.user_id = ANY(p_user_ids);
$$;

CREATE OR REPLACE FUNCTION public.upsert_user_interest_state(
    p_user_id TEXT,
    p_embedding VECTOR(300),
    p_weighted_sum VECTOR(300),
    p_total_weight DOUBLE PRECISION,
    p_interaction_count INTEGER,
    p_last_event_id BIGINT,
    p_profile_version TEXT,
    p_source_hash TEXT
) RETURNS VOID
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    INSERT INTO public.user_interest_embeddings (
        user_id, embedding, weighted_sum, total_weight, source_hash,
        profile_version, interaction_count, last_event_id, updated_at
    ) VALUES (
        p_user_id, p_embedding, p_weighted_sum, p_total_weight, p_source_hash,
        p_profile_version, p_interaction_count, p_last_event_id, now()
    )
    ON CONFLICT (user_id) DO UPDATE SET
        embedding = EXCLUDED.embedding,
        weighted_sum = EXCLUDED.weighted_sum,
        total_weight = EXCLUDED.total_weight,
        source_hash = EXCLUDED.source_hash,
        profile_version = EXCLUDED.profile_version,
        interaction_count = EXCLUDED.interaction_count,
        last_event_id = EXCLUDED.last_event_id,
        updated_at = now()
    WHERE EXCLUDED.last_event_id >= user_interest_embeddings.last_event_id;
$$;

CREATE OR REPLACE FUNCTION public.reset_user_interest_profiles(p_user_id TEXT)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    IF p_user_id IS NULL THEN
        DELETE FROM public.user_interest_embeddings;
    ELSE
        DELETE FROM public.user_interest_embeddings WHERE user_id = p_user_id;
    END IF;
END;
$$;

CREATE OR REPLACE FUNCTION public.list_post_embeddings_for_clustering(
    p_lookback_days INTEGER,
    p_embedding_model TEXT,
    p_embedding_version TEXT
) RETURNS TABLE (post_id TEXT, embedding VECTOR(300))
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT external_id, embedding
    FROM public.post_embeddings
    WHERE is_active = TRUE
      AND published_at >= now() - (p_lookback_days * interval '1 day')
      AND embedding_model = p_embedding_model
      AND embedding_version = p_embedding_version
    ORDER BY external_id;
$$;

CREATE OR REPLACE FUNCTION public.create_post_cluster_run(
    p_id UUID,
    p_algorithm_version TEXT,
    p_embedding_model TEXT,
    p_embedding_version TEXT,
    p_embedding_dimension INTEGER,
    p_k INTEGER,
    p_sample_size INTEGER,
    p_parameters JSONB
) RETURNS VOID
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    INSERT INTO public.post_cluster_runs (
        id, algorithm, algorithm_version, embedding_model, embedding_version,
        embedding_dimension, k, sample_size, parameters, status, started_at
    ) VALUES (
        p_id, 'kmeans', p_algorithm_version, p_embedding_model,
        p_embedding_version, p_embedding_dimension, p_k, p_sample_size,
        p_parameters, 'running', now()
    );
$$;

CREATE OR REPLACE FUNCTION public.validate_post_cluster_run(
    p_run_id UUID,
    p_k INTEGER,
    p_inertia DOUBLE PRECISION,
    p_silhouette_score DOUBLE PRECISION,
    p_parameters JSONB
) RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    UPDATE public.post_cluster_runs
    SET k = p_k,
        inertia = p_inertia,
        silhouette_score = p_silhouette_score,
        parameters = parameters || p_parameters,
        status = 'validated',
        completed_at = now(),
        error_message = NULL
    WHERE id = p_run_id AND status = 'running';
    IF NOT FOUND THEN
        RAISE EXCEPTION 'cluster run % no existe o no esta running', p_run_id;
    END IF;
END;
$$;

CREATE OR REPLACE FUNCTION public.fail_post_cluster_run(
    p_run_id UUID,
    p_error_message TEXT
) RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    UPDATE public.post_cluster_runs
    SET status = 'failed',
        completed_at = now(),
        error_message = LEFT(p_error_message, 4000)
    WHERE id = p_run_id AND status IN ('running', 'validated');
END;
$$;

CREATE OR REPLACE FUNCTION public.replace_post_cluster_artifacts(
    p_run_id UUID,
    p_clusters JSONB,
    p_memberships JSONB
) RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    DELETE FROM public.post_cluster_memberships WHERE run_id = p_run_id;
    DELETE FROM public.post_clusters WHERE run_id = p_run_id;

    IF NOT EXISTS (
        SELECT 1 FROM public.post_cluster_runs
        WHERE id = p_run_id AND status = 'validated'
    ) THEN
        RAISE EXCEPTION 'cluster run % no esta validado', p_run_id;
    END IF;

    INSERT INTO public.post_clusters (
        run_id, cluster_id, centroid, size, representative_post_ids
    )
    SELECT p_run_id, item.cluster_id, item.centroid::vector(300), item.size,
           COALESCE(item.representative_post_ids, '{}')
    FROM jsonb_to_recordset(p_clusters) AS item(
        cluster_id INTEGER, centroid TEXT, size INTEGER,
        representative_post_ids TEXT[]
    );

    INSERT INTO public.post_cluster_memberships (
        run_id, post_id, cluster_id, distance_to_centroid
    )
    SELECT p_run_id, item.post_id, item.cluster_id, item.distance
    FROM jsonb_to_recordset(p_memberships) AS item(
        post_id TEXT, cluster_id INTEGER, distance DOUBLE PRECISION
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.activate_post_cluster_run(p_run_id UUID)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM public.post_cluster_runs
        WHERE id = p_run_id AND status = 'active'
    ) THEN
        RETURN;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM public.post_cluster_runs
        WHERE id = p_run_id AND status IN ('validated', 'superseded')
    ) THEN
        RAISE EXCEPTION 'cluster run % no existe o no es activable', p_run_id;
    END IF;
    UPDATE public.post_cluster_runs SET status = 'superseded' WHERE status = 'active';
    UPDATE public.post_cluster_runs
    SET status = 'active', activated_at = now()
    WHERE id = p_run_id;
END;
$$;

CREATE OR REPLACE FUNCTION public.get_post_cluster_run(p_run_id UUID)
RETURNS TABLE (
    run_id UUID,
    status TEXT,
    k INTEGER,
    sample_size INTEGER,
    inertia DOUBLE PRECISION,
    silhouette_score DOUBLE PRECISION,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    activated_at TIMESTAMPTZ,
    error_message TEXT
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT id, post_cluster_runs.status, post_cluster_runs.k,
           post_cluster_runs.sample_size, post_cluster_runs.inertia,
           post_cluster_runs.silhouette_score, post_cluster_runs.started_at,
           post_cluster_runs.completed_at, post_cluster_runs.activated_at,
           post_cluster_runs.error_message
    FROM public.post_cluster_runs
    WHERE id = p_run_id;
$$;

CREATE OR REPLACE FUNCTION public.assign_posts_to_active_cluster(p_post_ids TEXT[])
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    affected INTEGER;
BEGIN
    WITH active_run AS (
        SELECT id, embedding_model, embedding_version
        FROM public.post_cluster_runs
        WHERE status = 'active'
        ORDER BY activated_at DESC NULLS LAST
        LIMIT 1
    ), assignments AS (
        SELECT active_run.id AS run_id,
               post.external_id AS post_id,
               nearest.cluster_id,
               nearest.distance
        FROM active_run
        JOIN public.post_embeddings post
          ON post.external_id = ANY(p_post_ids)
         AND post.is_active = TRUE
         AND post.embedding_model = active_run.embedding_model
         AND post.embedding_version = active_run.embedding_version
        CROSS JOIN LATERAL (
            SELECT cluster.cluster_id,
                   (post.embedding <=> cluster.centroid) AS distance
            FROM public.post_clusters cluster
            WHERE cluster.run_id = active_run.id
            ORDER BY post.embedding <=> cluster.centroid
            LIMIT 1
        ) nearest
    )
    INSERT INTO public.post_cluster_memberships (
        run_id, post_id, cluster_id, distance_to_centroid, assigned_at
    )
    SELECT run_id, post_id, cluster_id, distance, now()
    FROM assignments
    ON CONFLICT (run_id, post_id) DO UPDATE SET
        cluster_id = EXCLUDED.cluster_id,
        distance_to_centroid = EXCLUDED.distance_to_centroid,
        assigned_at = now();
    GET DIAGNOSTICS affected = ROW_COUNT;
    RETURN affected;
END;
$$;

CREATE OR REPLACE FUNCTION public.get_post_cluster_status()
RETURNS TABLE (
    active_run_id UUID,
    k INTEGER,
    sample_size INTEGER,
    silhouette_score DOUBLE PRECISION,
    activated_at TIMESTAMPTZ
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT id, post_cluster_runs.k, post_cluster_runs.sample_size,
           post_cluster_runs.silhouette_score, post_cluster_runs.activated_at
    FROM public.post_cluster_runs
    WHERE status = 'active'
    ORDER BY activated_at DESC NULLS LAST
    LIMIT 1;
$$;

-- Least-privilege grants when the expected roles exist.
REVOKE ALL ON FUNCTION public.upsert_post_embedding(TEXT, TEXT, JSONB, VECTOR, TEXT, TEXT, TEXT, BOOLEAN, TEXT, TEXT, TIMESTAMPTZ, BIGINT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.deactivate_post_embedding(TEXT, BIGINT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_post_sync_checkpoint(TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.save_post_sync_checkpoint(TEXT, BIGINT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.reset_post_sync_checkpoint(TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_post_feed_features(TEXT, TEXT[]) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_post_embeddings_for_profile(TEXT[]) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_user_interest_states(TEXT[]) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.upsert_user_interest_state(TEXT, VECTOR, VECTOR, DOUBLE PRECISION, INTEGER, BIGINT, TEXT, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.reset_user_interest_profiles(TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.list_post_embeddings_for_clustering(INTEGER, TEXT, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.create_post_cluster_run(UUID, TEXT, TEXT, TEXT, INTEGER, INTEGER, INTEGER, JSONB) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.validate_post_cluster_run(UUID, INTEGER, DOUBLE PRECISION, DOUBLE PRECISION, JSONB) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.fail_post_cluster_run(UUID, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.replace_post_cluster_artifacts(UUID, JSONB, JSONB) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.activate_post_cluster_run(UUID) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.assign_posts_to_active_cluster(TEXT[]) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_post_cluster_status() FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_post_cluster_run(UUID) FROM PUBLIC;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_reader') THEN
        GRANT EXECUTE ON FUNCTION public.get_post_feed_features(TEXT, TEXT[]) TO nlp_reader;
        GRANT EXECUTE ON FUNCTION public.get_post_cluster_status() TO nlp_reader;
        GRANT EXECUTE ON FUNCTION public.get_post_cluster_run(UUID) TO nlp_reader;
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_writer') THEN
        GRANT EXECUTE ON FUNCTION public.upsert_post_embedding(TEXT, TEXT, JSONB, VECTOR, TEXT, TEXT, TEXT, BOOLEAN, TEXT, TEXT, TIMESTAMPTZ, BIGINT) TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.deactivate_post_embedding(TEXT, BIGINT) TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.get_post_sync_checkpoint(TEXT) TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.save_post_sync_checkpoint(TEXT, BIGINT) TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.reset_post_sync_checkpoint(TEXT) TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.get_post_embeddings_for_profile(TEXT[]) TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.get_user_interest_states(TEXT[]) TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.upsert_user_interest_state(TEXT, VECTOR, VECTOR, DOUBLE PRECISION, INTEGER, BIGINT, TEXT, TEXT) TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.reset_user_interest_profiles(TEXT) TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.list_post_embeddings_for_clustering(INTEGER, TEXT, TEXT) TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.create_post_cluster_run(UUID, TEXT, TEXT, TEXT, INTEGER, INTEGER, INTEGER, JSONB) TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.validate_post_cluster_run(UUID, INTEGER, DOUBLE PRECISION, DOUBLE PRECISION, JSONB) TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.fail_post_cluster_run(UUID, TEXT) TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.replace_post_cluster_artifacts(UUID, JSONB, JSONB) TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.activate_post_cluster_run(UUID) TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.assign_posts_to_active_cluster(TEXT[]) TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.get_post_cluster_status() TO nlp_writer;
        GRANT EXECUTE ON FUNCTION public.get_post_cluster_run(UUID) TO nlp_writer;
    END IF;
END $$;

COMMIT;


BEGIN;

DO $$
DECLARE
    actual_type TEXT;
BEGIN
    SELECT format_type(attribute.atttypid, attribute.atttypmod)
    INTO actual_type
    FROM pg_attribute attribute
    WHERE attribute.attrelid = 'public.post_embeddings'::regclass
      AND attribute.attname = 'embedding'
      AND NOT attribute.attisdropped;
    IF actual_type IS DISTINCT FROM 'vector(300)' THEN
        RAISE EXCEPTION 'post_embeddings.embedding debe ser vector(300), actual=%', actual_type;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_owner')
       OR NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_reader')
       OR NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_writer') THEN
        RAISE EXCEPTION 'se requieren los roles nlp_owner, nlp_reader y nlp_writer';
    END IF;
END $$;

CREATE OR REPLACE FUNCTION public.upsert_post_embedding(
    p_external_id TEXT,
    p_document TEXT,
    p_metadata JSONB,
    p_embedding VECTOR(300),
    p_content_hash TEXT,
    p_embedding_model TEXT,
    p_embedding_version TEXT,
    p_is_active BOOLEAN,
    p_author_type TEXT,
    p_author_id TEXT,
    p_published_at TIMESTAMPTZ,
    p_source_version BIGINT
) RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    current_version BIGINT;
    current_hash TEXT;
BEGIN
    IF p_source_version IS NULL OR p_source_version < 0 THEN
        RAISE EXCEPTION 'source_version es obligatorio y no negativo para %', p_external_id;
    END IF;

    SELECT source_version, content_hash
    INTO current_version, current_hash
    FROM public.post_embeddings
    WHERE external_id = p_external_id;

    IF FOUND AND current_version IS NOT NULL THEN
        IF current_version > p_source_version THEN
            RETURN;
        END IF;
        IF current_version = p_source_version THEN
            IF current_hash = p_content_hash THEN
                RETURN;
            END IF;
            RAISE EXCEPTION 'conflicto: misma source_version con contenido distinto para %', p_external_id;
        END IF;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM public.post_embedding_tombstones tombstone
        WHERE tombstone.post_id = p_external_id
          AND tombstone.source_version >= p_source_version
    ) THEN
        RETURN;
    END IF;

    DELETE FROM public.post_embedding_tombstones
    WHERE post_id = p_external_id AND source_version < p_source_version;

    INSERT INTO public.post_embeddings (
        external_id, document, metadata, embedding, content_hash,
        embedding_model, embedding_version, is_active, author_type,
        author_id, published_at, source_version, indexed_at, updated_at
    ) VALUES (
        p_external_id, p_document, p_metadata, p_embedding, p_content_hash,
        p_embedding_model, p_embedding_version, p_is_active, p_author_type,
        p_author_id, p_published_at, p_source_version, now(), now()
    )
    ON CONFLICT (external_id) DO UPDATE SET
        document = EXCLUDED.document,
        metadata = EXCLUDED.metadata,
        embedding = EXCLUDED.embedding,
        content_hash = EXCLUDED.content_hash,
        embedding_model = EXCLUDED.embedding_model,
        embedding_version = EXCLUDED.embedding_version,
        is_active = EXCLUDED.is_active,
        author_type = EXCLUDED.author_type,
        author_id = EXCLUDED.author_id,
        published_at = EXCLUDED.published_at,
        source_version = EXCLUDED.source_version,
        indexed_at = now(),
        updated_at = now()
    WHERE post_embeddings.source_version IS NULL
       OR EXCLUDED.source_version > post_embeddings.source_version;
END;
$$;

CREATE OR REPLACE FUNCTION public.deactivate_post_embedding(
    p_external_id TEXT,
    p_source_version BIGINT
) RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    IF p_source_version IS NULL OR p_source_version < 0 THEN
        RAISE EXCEPTION 'source_version es obligatorio y no negativo para %', p_external_id;
    END IF;
    INSERT INTO public.post_embedding_tombstones (post_id, source_version, deleted_at)
    VALUES (p_external_id, p_source_version, now())
    ON CONFLICT (post_id) DO UPDATE SET
        source_version = GREATEST(post_embedding_tombstones.source_version, EXCLUDED.source_version),
        deleted_at = CASE
            WHEN EXCLUDED.source_version >= post_embedding_tombstones.source_version THEN now()
            ELSE post_embedding_tombstones.deleted_at
        END;
    UPDATE public.post_embeddings
    SET is_active = FALSE,
        source_version = GREATEST(COALESCE(source_version, 0), p_source_version),
        indexed_at = now(),
        updated_at = now()
    WHERE external_id = p_external_id
      AND (source_version IS NULL OR p_source_version >= source_version);
END;
$$;

CREATE OR REPLACE FUNCTION public.get_post_content_hashes(p_external_ids TEXT[])
RETURNS TABLE (external_id TEXT, content_hash TEXT)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT post.external_id, post.content_hash
    FROM public.post_embeddings post
    WHERE post.external_id = ANY(p_external_ids)
      AND post.is_active = TRUE
      AND NOT EXISTS (
          SELECT 1 FROM public.post_embedding_tombstones tombstone
          WHERE tombstone.post_id = post.external_id
            AND tombstone.source_version >= COALESCE(post.source_version, 0)
      );
$$;

CREATE OR REPLACE FUNCTION public.get_post_feed_features(
    p_user_id TEXT,
    p_post_ids TEXT[]
) RETURNS TABLE (
    external_id TEXT,
    semantic_similarity DOUBLE PRECISION,
    cluster_id INTEGER,
    cluster_run_id TEXT,
    cold_start BOOLEAN,
    post_embedding TEXT
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    WITH requested AS (
        SELECT post_id, ordinality
        FROM unnest(p_post_ids) WITH ORDINALITY AS ids(post_id, ordinality)
    ), active_run AS (
        SELECT id FROM public.post_cluster_runs
        WHERE status = 'active'
        ORDER BY activated_at DESC NULLS LAST
        LIMIT 1
    ), profile AS (
        SELECT embedding FROM public.user_interest_embeddings WHERE user_id = p_user_id
    )
    SELECT requested.post_id,
           CASE WHEN profile.embedding IS NULL OR post.embedding IS NULL THEN 0.0
                ELSE 1 - (post.embedding <=> profile.embedding) END,
           membership.cluster_id,
           active_run.id::text,
           (profile.embedding IS NULL OR post.embedding IS NULL),
           post.embedding::text
    FROM requested
    LEFT JOIN public.post_embeddings post
      ON post.external_id = requested.post_id AND post.is_active = TRUE
    LEFT JOIN profile ON TRUE
    LEFT JOIN active_run ON TRUE
    LEFT JOIN public.post_cluster_memberships membership
      ON membership.post_id = post.external_id AND membership.run_id = active_run.id
    ORDER BY requested.ordinality;
$$;

CREATE OR REPLACE FUNCTION public.activate_post_cluster_run(p_run_id UUID)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    expected_k INTEGER;
    expected_sample_size INTEGER;
    cluster_count INTEGER;
    membership_count INTEGER;
    declared_size INTEGER;
BEGIN
    PERFORM pg_advisory_xact_lock(hashtext('frimeet-post-cluster-activation'));
    SELECT k, sample_size INTO expected_k, expected_sample_size
    FROM public.post_cluster_runs
    WHERE id = p_run_id AND status IN ('validated', 'superseded', 'active')
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'cluster run % no existe o no es activable', p_run_id;
    END IF;
    SELECT count(*), COALESCE(sum(size), 0)
    INTO cluster_count, declared_size
    FROM public.post_clusters WHERE run_id = p_run_id;
    SELECT count(*) INTO membership_count
    FROM public.post_cluster_memberships WHERE run_id = p_run_id;
    IF cluster_count <> expected_k
       OR membership_count <> expected_sample_size
       OR declared_size <> expected_sample_size THEN
        RAISE EXCEPTION 'cluster run % tiene artefactos incompletos: clusters=%/% memberships=%/% size=%/%',
            p_run_id, cluster_count, expected_k, membership_count,
            expected_sample_size, declared_size, expected_sample_size;
    END IF;
    IF EXISTS (SELECT 1 FROM public.post_cluster_runs WHERE id = p_run_id AND status = 'active') THEN
        RETURN;
    END IF;
    UPDATE public.post_cluster_runs SET status = 'superseded' WHERE status = 'active';
    UPDATE public.post_cluster_runs
    SET status = 'active', activated_at = now()
    WHERE id = p_run_id;
END;
$$;

ALTER FUNCTION public.upsert_post_embedding(text, text, jsonb, vector, text, text, text, boolean, text, text, timestamptz, bigint) OWNER TO nlp_owner;
ALTER FUNCTION public.deactivate_post_embedding(text, bigint) OWNER TO nlp_owner;
ALTER FUNCTION public.get_post_content_hashes(text[]) OWNER TO nlp_owner;
ALTER FUNCTION public.get_post_feed_features(text, text[]) OWNER TO nlp_owner;
ALTER FUNCTION public.activate_post_cluster_run(uuid) OWNER TO nlp_owner;

REVOKE ALL ON FUNCTION public.upsert_post_embedding(text, text, jsonb, vector, text, text, text, boolean, text, text, timestamptz, bigint) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.deactivate_post_embedding(text, bigint) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_post_content_hashes(text[]) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_post_feed_features(text, text[]) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.activate_post_cluster_run(uuid) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION public.get_post_feed_features(text, text[]) TO nlp_reader;
GRANT EXECUTE ON FUNCTION public.upsert_post_embedding(text, text, jsonb, vector, text, text, text, boolean, text, text, timestamptz, bigint) TO nlp_writer;
GRANT EXECUTE ON FUNCTION public.deactivate_post_embedding(text, bigint) TO nlp_writer;
GRANT EXECUTE ON FUNCTION public.get_post_content_hashes(text[]) TO nlp_writer;
GRANT EXECUTE ON FUNCTION public.activate_post_cluster_run(uuid) TO nlp_writer;

COMMIT;


-- Reconciliacion final de propietarios y minimo privilegio.
DO $security$
DECLARE item RECORD;
BEGIN
    FOR item IN
        SELECT format('%I.%I', table_schema, table_name) AS object_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = ANY(ARRAY[
              'place_embeddings','post_embeddings','user_embeddings',
              'club_embeddings','group_embeddings','event_embeddings',
              'post_cluster_runs','post_clusters','post_cluster_memberships',
              'user_interest_embeddings','post_sync_checkpoints','place_sync_checkpoints',
              'post_embedding_tombstones'
          ])
    LOOP
        EXECUTE format('ALTER TABLE %s OWNER TO nlp_owner', item.object_name);
        EXECUTE format('REVOKE ALL ON TABLE %s FROM PUBLIC', item.object_name);
    END LOOP;

    FOR item IN
        SELECT procedure.oid::regprocedure AS signature
        FROM pg_proc procedure
        JOIN pg_namespace namespace ON namespace.oid = procedure.pronamespace
        WHERE namespace.nspname = 'public'
          AND procedure.proname = ANY(ARRAY[
              'match_places','match_posts','upsert_place_embedding','deactivate_place_embedding',
              'upsert_post_embedding','get_place_content_hashes',
              'get_post_content_hashes','search_resource_embeddings',
              'get_resource_content_hashes','upsert_resource_embedding',
               'deactivate_post_embedding','get_post_sync_checkpoint',
               'save_post_sync_checkpoint','reset_post_sync_checkpoint',
               'get_place_sync_checkpoint','save_place_sync_checkpoint',
              'get_post_feed_features','get_post_embeddings_for_profile',
              'get_user_interest_states','upsert_user_interest_state',
              'reset_user_interest_profiles','create_post_cluster_run',
              'validate_post_cluster_run','replace_post_cluster_artifacts',
              'fail_post_cluster_run','get_post_cluster_status',
              'get_post_cluster_run','activate_post_cluster_run',
              'assign_posts_to_active_cluster','list_post_embeddings_for_clustering'
          ])
    LOOP
        EXECUTE format('ALTER FUNCTION %s OWNER TO nlp_owner', item.signature);
        EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC', item.signature);
    END LOOP;
END
$security$;

GRANT USAGE ON SCHEMA public TO nlp_reader, nlp_writer;
GRANT EXECUTE ON FUNCTION public.match_places(vector, integer, jsonb) TO nlp_reader;
GRANT EXECUTE ON FUNCTION public.match_posts(vector, integer, jsonb) TO nlp_reader;
GRANT EXECUTE ON FUNCTION public.search_resource_embeddings(text, text, vector, integer, jsonb) TO nlp_reader;
GRANT EXECUTE ON FUNCTION public.get_post_feed_features(text, text[]) TO nlp_reader;

GRANT EXECUTE ON FUNCTION public.upsert_place_embedding(text, text, jsonb, vector, text, text, text, boolean) TO nlp_writer;
GRANT EXECUTE ON FUNCTION public.deactivate_place_embedding(text) TO nlp_writer;
GRANT EXECUTE ON FUNCTION public.upsert_post_embedding(text, text, jsonb, vector, text, text, text, boolean, text, text, timestamptz, bigint) TO nlp_writer;
GRANT EXECUTE ON FUNCTION public.get_place_content_hashes(text[]) TO nlp_writer;
GRANT EXECUTE ON FUNCTION public.get_post_content_hashes(text[]) TO nlp_writer;
GRANT EXECUTE ON FUNCTION public.get_resource_content_hashes(text, text[]) TO nlp_writer;
GRANT EXECUTE ON FUNCTION public.upsert_resource_embedding(text, text, text, jsonb, vector, text, text, text, boolean) TO nlp_writer;
GRANT EXECUTE ON FUNCTION public.get_place_sync_checkpoint(text) TO nlp_writer;
GRANT EXECUTE ON FUNCTION public.save_place_sync_checkpoint(text, bigint) TO nlp_writer;

-- Configura contrasenas fuera de este archivo:
-- ALTER ROLE nlp_reader PASSWORD '<secreto-reader>';
-- ALTER ROLE nlp_writer PASSWORD '<secreto-writer>';

-- El retriever semantico de Places se despliega de forma aditiva despues de
-- este esquema base. Ejecuta, en orden:
--   sql/migrations/20260716_02_places_semantic_v1.sql
--   sql/migrations/20260721_03_place_facets_and_incremental_sync.sql
