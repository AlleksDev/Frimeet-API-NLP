-- Ejecutar desde pgAdmin despues de recargar lugares con weighted-tags-v2.

SELECT
    count(*) AS total_places,
    count(*) FILTER (
        WHERE metadata->>'semantic_document_version' = 'weighted-tags-v2'
    ) AS weighted_v2_places,
    min(vector_dims(embedding)) AS min_dimensions,
    max(vector_dims(embedding)) AS max_dimensions,
    min(embedding_model) AS embedding_model,
    min(embedding_version) AS embedding_version,
    count(*) FILTER (
        WHERE COALESCE(metadata->>'tags', '') <> ''
    ) AS places_with_resolved_tags,
    count(*) FILTER (
        WHERE metadata ? 'unknown_tag_ids'
    ) AS places_with_unknown_tag_ids
FROM public.place_embeddings;

SELECT
    metadata->>'category' AS category,
    count(*) AS places
FROM public.place_embeddings
WHERE is_active = true
GROUP BY metadata->>'category'
ORDER BY places DESC;

SELECT
    external_id,
    metadata->>'name' AS name,
    metadata->>'category' AS category,
    metadata->>'tags' AS resolved_tags,
    metadata->'tag_ids' AS original_tag_ids,
    metadata->'unknown_tag_ids' AS unknown_tag_ids,
    left(document, 300) AS weighted_document_preview
FROM public.place_embeddings
ORDER BY updated_at DESC
LIMIT 20;

