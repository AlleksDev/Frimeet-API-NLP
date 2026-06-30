-- Ejecutar después de recargar lugares y posts con los jobs E5/SBERT.

SELECT
    'place_embeddings' AS table_name,
    count(*) AS total_rows,
    min(vector_dims(embedding)) AS min_dimension,
    max(vector_dims(embedding)) AS max_dimension,
    min(embedding_model) AS min_model,
    max(embedding_model) AS max_model,
    min(embedding_version) AS min_version,
    max(embedding_version) AS max_version,
    round(avg(vector_norm(embedding))::numeric, 6) AS mean_norm
FROM public.place_embeddings
UNION ALL
SELECT
    'post_embeddings' AS table_name,
    count(*) AS total_rows,
    min(vector_dims(embedding)) AS min_dimension,
    max(vector_dims(embedding)) AS max_dimension,
    min(embedding_model) AS min_model,
    max(embedding_model) AS max_model,
    min(embedding_version) AS min_version,
    max(embedding_version) AS max_version,
    round(avg(vector_norm(embedding))::numeric, 6) AS mean_norm
FROM public.post_embeddings;

SELECT
    embedding_model,
    embedding_version,
    count(*) AS rows
FROM (
    SELECT embedding_model, embedding_version FROM public.place_embeddings
    UNION ALL
    SELECT embedding_model, embedding_version FROM public.post_embeddings
) embeddings
GROUP BY embedding_model, embedding_version
ORDER BY rows DESC;

SELECT
    count(*) AS place_rows,
    count(*) FILTER (
        WHERE metadata->>'semantic_document_version' = 'weighted-tags-v2'
    ) AS weighted_v2_rows
FROM public.place_embeddings;
