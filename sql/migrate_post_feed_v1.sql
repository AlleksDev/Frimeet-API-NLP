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
