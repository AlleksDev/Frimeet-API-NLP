-- Run after sql/migrate_post_feed_v1.sql. Every boolean must be true.
SELECT
    to_regclass('public.post_cluster_runs') IS NOT NULL AS cluster_runs_exists,
    to_regclass('public.post_clusters') IS NOT NULL AS clusters_exists,
    to_regclass('public.post_cluster_memberships') IS NOT NULL AS memberships_exists,
    to_regclass('public.user_interest_embeddings') IS NOT NULL AS profiles_exists,
    to_regclass('public.post_sync_checkpoints') IS NOT NULL AS checkpoints_exists,
    to_regclass('public.post_embedding_tombstones') IS NOT NULL AS tombstones_exists;

SELECT count(*) = 5 AS post_embedding_columns_complete
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'post_embeddings'
  AND column_name IN ('author_type', 'author_id', 'published_at', 'source_version', 'indexed_at');

SELECT count(*) <= 1 AS at_most_one_active_cluster_run
FROM public.post_cluster_runs
WHERE status = 'active';

SELECT
    to_regprocedure('public.upsert_post_embedding(text,text,jsonb,vector,text,text,text,boolean,text,text,timestamp with time zone,bigint)') IS NOT NULL AS upsert_exists,
    to_regprocedure('public.deactivate_post_embedding(text,bigint)') IS NOT NULL AS deactivate_exists,
    to_regprocedure('public.get_post_feed_features(text,text[])') IS NOT NULL AS feed_features_exists,
    to_regprocedure('public.get_post_sync_checkpoint(text)') IS NOT NULL AS checkpoint_read_exists,
    to_regprocedure('public.save_post_sync_checkpoint(text,bigint)') IS NOT NULL AS checkpoint_write_exists,
    to_regprocedure('public.reset_post_sync_checkpoint(text)') IS NOT NULL AS checkpoint_reset_exists,
    to_regprocedure('public.get_user_interest_states(text[])') IS NOT NULL AS profile_read_exists,
    to_regprocedure('public.upsert_user_interest_state(text,vector,vector,double precision,integer,bigint,text,text)') IS NOT NULL AS profile_write_exists,
    to_regprocedure('public.reset_user_interest_profiles(text)') IS NOT NULL AS profile_reset_exists,
    to_regprocedure('public.list_post_embeddings_for_clustering(integer,text,text)') IS NOT NULL AS cluster_source_exists,
    to_regprocedure('public.create_post_cluster_run(uuid,text,text,text,integer,integer,integer,jsonb)') IS NOT NULL AS cluster_create_exists,
    to_regprocedure('public.validate_post_cluster_run(uuid,integer,double precision,double precision,jsonb)') IS NOT NULL AS cluster_validate_exists,
    to_regprocedure('public.fail_post_cluster_run(uuid,text)') IS NOT NULL AS cluster_fail_exists,
    to_regprocedure('public.activate_post_cluster_run(uuid)') IS NOT NULL AS cluster_activation_exists,
    to_regprocedure('public.get_post_cluster_run(uuid)') IS NOT NULL AS cluster_detail_exists;
SELECT to_regprocedure('public.assign_posts_to_active_cluster(text[])') IS NOT NULL
       AS new_post_assignment_exists;

SELECT
    CASE WHEN EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_reader')
         THEN has_function_privilege('nlp_reader', 'public.get_post_feed_features(text,text[])', 'EXECUTE')
         ELSE true END AS reader_feed_permission,
    CASE WHEN EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_writer')
         THEN has_function_privilege('nlp_writer', 'public.deactivate_post_embedding(text,bigint)', 'EXECUTE')
         ELSE true END AS writer_sync_permission;

SELECT count(*) AS memberships_without_active_post
FROM public.post_cluster_memberships membership
LEFT JOIN public.post_embeddings post ON post.external_id = membership.post_id
WHERE post.external_id IS NULL;
