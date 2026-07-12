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
