BEGIN;

-- Rollback lógico: restaura únicamente las funciones reemplazadas por V2.
-- No elimina tablas, perfiles, clusters ni embeddings.
CREATE OR REPLACE FUNCTION public.get_post_content_hashes(p_external_ids TEXT[])
RETURNS TABLE (external_id TEXT, content_hash TEXT)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public
AS $$
    SELECT post.external_id, post.content_hash
    FROM public.post_embeddings post
    WHERE post.external_id = ANY(p_external_ids);
$$;

CREATE OR REPLACE FUNCTION public.get_post_feed_features(p_user_id TEXT, p_post_ids TEXT[])
RETURNS TABLE (external_id TEXT, semantic_similarity DOUBLE PRECISION, cluster_id INTEGER, cluster_run_id TEXT, cold_start BOOLEAN, post_embedding TEXT)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public
AS $$
    WITH requested AS (
        SELECT post_id, ordinality FROM unnest(p_post_ids) WITH ORDINALITY AS ids(post_id, ordinality)
    ), active_run AS (
        SELECT id FROM public.post_cluster_runs WHERE status = 'active' ORDER BY activated_at DESC NULLS LAST LIMIT 1
    ), profile AS (
        SELECT embedding FROM public.user_interest_embeddings WHERE user_id = p_user_id
    )
    SELECT requested.post_id,
           CASE WHEN profile.embedding IS NULL OR post.embedding IS NULL THEN 0.0 ELSE 1 - (post.embedding <=> profile.embedding) END,
           membership.cluster_id, active_run.id::text, profile.embedding IS NULL, post.embedding::text
    FROM requested
    LEFT JOIN public.post_embeddings post ON post.external_id = requested.post_id AND post.is_active = TRUE
    LEFT JOIN profile ON TRUE LEFT JOIN active_run ON TRUE
    LEFT JOIN public.post_cluster_memberships membership ON membership.post_id = post.external_id AND membership.run_id = active_run.id
    ORDER BY requested.ordinality;
$$;

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

-- Se conserva el endurecimiento de permisos incluso al volver a la semántica V1.
ALTER FUNCTION public.get_post_content_hashes(text[]) OWNER TO nlp_owner;
ALTER FUNCTION public.get_post_feed_features(text, text[]) OWNER TO nlp_owner;
REVOKE ALL ON FUNCTION public.get_post_content_hashes(text[]) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_post_feed_features(text, text[]) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.get_post_content_hashes(text[]) TO nlp_writer;
GRANT EXECUTE ON FUNCTION public.get_post_feed_features(text, text[]) TO nlp_reader;

COMMIT;
