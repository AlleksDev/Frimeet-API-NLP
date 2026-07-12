SELECT
    format_type(attribute.atttypid, attribute.atttypmod) = 'vector(300)' AS post_embedding_dimension_ok,
    to_regprocedure('public.upsert_post_embedding(text,text,jsonb,vector,text,text,text,boolean,text,text,timestamp with time zone,bigint)') IS NOT NULL AS upsert_exists,
    to_regprocedure('public.deactivate_post_embedding(text,bigint)') IS NOT NULL AS deactivate_exists,
    to_regprocedure('public.get_post_feed_features(text,text[])') IS NOT NULL AS feed_features_exists,
    to_regprocedure('public.activate_post_cluster_run(uuid)') IS NOT NULL AS activate_exists,
    pg_get_userbyid(procedure.proowner) = 'nlp_owner' AS upsert_owned_by_nlp_owner,
    has_function_privilege('nlp_reader', 'public.get_post_feed_features(text,text[])', 'EXECUTE') AS reader_can_rank,
    has_function_privilege('nlp_writer', 'public.upsert_post_embedding(text,text,jsonb,vector,text,text,text,boolean,text,text,timestamp with time zone,bigint)', 'EXECUTE') AS writer_can_upsert,
    NOT EXISTS (
        SELECT 1
        FROM pg_proc checked
        CROSS JOIN LATERAL aclexplode(COALESCE(checked.proacl, acldefault('f', checked.proowner))) privilege
        WHERE checked.oid = 'public.get_post_feed_features(text,text[])'::regprocedure
          AND privilege.grantee = 0 AND privilege.privilege_type = 'EXECUTE'
    ) AS public_cannot_rank,
    NOT EXISTS (
        SELECT 1
        FROM pg_proc checked
        CROSS JOIN LATERAL aclexplode(COALESCE(checked.proacl, acldefault('f', checked.proowner))) privilege
        WHERE checked.oid = 'public.upsert_post_embedding(text,text,jsonb,vector,text,text,text,boolean,text,text,timestamp with time zone,bigint)'::regprocedure
          AND privilege.grantee = 0 AND privilege.privilege_type = 'EXECUTE'
    ) AS public_cannot_upsert,
    (SELECT count(*) FROM public.post_embeddings WHERE source_version IS NULL) AS posts_without_source_version,
    (SELECT count(*) FROM public.post_cluster_runs run
     WHERE run.status = 'active'
       AND ((SELECT count(*) FROM public.post_clusters cluster WHERE cluster.run_id = run.id) <> run.k
         OR (SELECT count(*) FROM public.post_cluster_memberships membership WHERE membership.run_id = run.id) <> run.sample_size)) AS invalid_active_runs,
    (SELECT count(*) FROM public.post_embeddings post
     JOIN public.post_embedding_tombstones tombstone ON tombstone.post_id = post.external_id
     WHERE post.is_active = TRUE
       AND tombstone.source_version >= COALESCE(post.source_version, 0)) AS active_posts_blocked_by_tombstone
FROM pg_attribute attribute
JOIN pg_proc procedure
  ON procedure.oid = 'public.upsert_post_embedding(text,text,jsonb,vector,text,text,text,boolean,text,text,timestamp with time zone,bigint)'::regprocedure
WHERE attribute.attrelid = 'public.post_embeddings'::regclass
  AND attribute.attname = 'embedding'
  AND NOT attribute.attisdropped;
