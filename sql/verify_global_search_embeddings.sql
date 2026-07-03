-- Read-only verification after syncing users, clubs, groups and events.
WITH indexed_resources AS (
    SELECT 'users' AS resource_type, embedding, embedding_model, embedding_version, is_active
    FROM user_embeddings
    UNION ALL
    SELECT 'clubs', embedding, embedding_model, embedding_version, is_active
    FROM club_embeddings
    UNION ALL
    SELECT 'groups', embedding, embedding_model, embedding_version, is_active
    FROM group_embeddings
    UNION ALL
    SELECT 'events', embedding, embedding_model, embedding_version, is_active
    FROM event_embeddings
)
SELECT
    resource_type,
    count(*) AS indexed_count,
    count(*) FILTER (WHERE is_active) AS active_count,
    min(vector_dims(embedding)) AS min_dimension,
    max(vector_dims(embedding)) AS max_dimension,
    min(embedding_model) AS embedding_model,
    min(embedding_version) AS embedding_version,
    round(avg(vector_norm(embedding))::numeric, 4) AS average_norm
FROM indexed_resources
GROUP BY resource_type
ORDER BY resource_type;

SELECT
    to_regprocedure(
        'search_resource_embeddings(text,text,vector,integer,jsonb)'
    ) IS NOT NULL AS hybrid_search_function_exists,
    to_regprocedure(
        'get_resource_content_hashes(text,text[])'
    ) IS NOT NULL AS content_hash_function_exists,
    to_regprocedure(
        'upsert_resource_embedding(text,text,text,jsonb,vector,text,text,text,boolean)'
    ) IS NOT NULL AS upsert_function_exists;
