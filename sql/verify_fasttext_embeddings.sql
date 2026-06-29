SELECT
    'place_embeddings' AS table_name,
    count(*) AS row_count,
    min(vector_dims(embedding)) AS min_dimensions,
    max(vector_dims(embedding)) AS max_dimensions,
    min(embedding_model) AS embedding_model,
    min(embedding_version) AS embedding_version,
    min(vector_norm(embedding)) AS min_norm,
    max(vector_norm(embedding)) AS max_norm
FROM public.place_embeddings
UNION ALL
SELECT
    'post_embeddings' AS table_name,
    count(*) AS row_count,
    min(vector_dims(embedding)) AS min_dimensions,
    max(vector_dims(embedding)) AS max_dimensions,
    min(embedding_model) AS embedding_model,
    min(embedding_version) AS embedding_version,
    min(vector_norm(embedding)) AS min_norm,
    max(vector_norm(embedding)) AS max_norm
FROM public.post_embeddings;
