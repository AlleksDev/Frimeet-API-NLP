BEGIN;

DROP FUNCTION IF EXISTS public.get_post_cluster_status();
DROP FUNCTION IF EXISTS public.get_post_cluster_run(UUID);
DROP FUNCTION IF EXISTS public.activate_post_cluster_run(UUID);
DROP FUNCTION IF EXISTS public.assign_posts_to_active_cluster(TEXT[]);
DROP FUNCTION IF EXISTS public.replace_post_cluster_artifacts(UUID, JSONB, JSONB);
DROP FUNCTION IF EXISTS public.fail_post_cluster_run(UUID, TEXT);
DROP FUNCTION IF EXISTS public.validate_post_cluster_run(UUID, INTEGER, DOUBLE PRECISION, DOUBLE PRECISION, JSONB);
DROP FUNCTION IF EXISTS public.create_post_cluster_run(UUID, TEXT, TEXT, TEXT, INTEGER, INTEGER, INTEGER, JSONB);
DROP FUNCTION IF EXISTS public.list_post_embeddings_for_clustering(INTEGER, TEXT, TEXT);
DROP FUNCTION IF EXISTS public.upsert_user_interest_state(TEXT, VECTOR, VECTOR, DOUBLE PRECISION, INTEGER, BIGINT, TEXT, TEXT);
DROP FUNCTION IF EXISTS public.reset_user_interest_profiles(TEXT);
DROP FUNCTION IF EXISTS public.get_user_interest_states(TEXT[]);
DROP FUNCTION IF EXISTS public.get_post_embeddings_for_profile(TEXT[]);
DROP FUNCTION IF EXISTS public.get_post_feed_features(TEXT, TEXT[]);
DROP FUNCTION IF EXISTS public.save_post_sync_checkpoint(TEXT, BIGINT);
DROP FUNCTION IF EXISTS public.reset_post_sync_checkpoint(TEXT);
DROP FUNCTION IF EXISTS public.get_post_sync_checkpoint(TEXT);
DROP FUNCTION IF EXISTS public.deactivate_post_embedding(TEXT, BIGINT);
DROP FUNCTION IF EXISTS public.upsert_post_embedding(TEXT, TEXT, JSONB, VECTOR, TEXT, TEXT, TEXT, BOOLEAN, TEXT, TEXT, TIMESTAMPTZ, BIGINT);

DROP TABLE IF EXISTS public.post_sync_checkpoints;
DROP TABLE IF EXISTS public.post_embedding_tombstones;
DROP TABLE IF EXISTS public.user_interest_embeddings;
DROP TABLE IF EXISTS public.post_cluster_memberships;
DROP TABLE IF EXISTS public.post_clusters;
DROP TABLE IF EXISTS public.post_cluster_runs;

DROP INDEX IF EXISTS public.post_embeddings_author_idx;
DROP INDEX IF EXISTS public.post_embeddings_active_published_idx;

ALTER TABLE public.post_embeddings
    DROP COLUMN IF EXISTS author_type,
    DROP COLUMN IF EXISTS author_id,
    DROP COLUMN IF EXISTS published_at,
    DROP COLUMN IF EXISTS source_version,
    DROP COLUMN IF EXISTS indexed_at;

CREATE OR REPLACE FUNCTION public.upsert_post_embedding(
    p_external_id TEXT,
    p_document TEXT,
    p_metadata JSONB,
    p_embedding VECTOR(300),
    p_content_hash TEXT,
    p_embedding_model TEXT,
    p_embedding_version TEXT,
    p_is_active BOOLEAN
) RETURNS VOID
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    INSERT INTO public.post_embeddings (
        external_id, document, metadata, embedding, content_hash,
        embedding_model, embedding_version, is_active, updated_at
    ) VALUES (
        p_external_id, p_document, p_metadata, p_embedding, p_content_hash,
        p_embedding_model, p_embedding_version, p_is_active, now()
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

COMMIT;
