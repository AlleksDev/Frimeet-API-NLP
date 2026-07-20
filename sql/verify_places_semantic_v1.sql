BEGIN;
SET TRANSACTION READ ONLY;

DO $$
DECLARE
    embedding_type TEXT;
    embedding_count BIGINT;
    active_count BIGINT;
    profile_count BIGINT;
    invalid_norm_count BIGINT;
BEGIN
    SELECT format_type(attribute.atttypid, attribute.atttypmod)
    INTO embedding_type
    FROM pg_attribute AS attribute
    WHERE attribute.attrelid = 'public.place_embeddings_semantic_v1'::regclass
      AND attribute.attname = 'embedding'
      AND NOT attribute.attisdropped;

    IF embedding_type IS DISTINCT FROM 'vector(768)' THEN
        RAISE EXCEPTION 'Expected place_embeddings_semantic_v1.embedding vector(768), got %', embedding_type;
    END IF;
    IF to_regprocedure('public.match_places_semantic_v1(vector,integer,jsonb)') IS NULL THEN
        RAISE EXCEPTION 'match_places_semantic_v1 is missing';
    END IF;
    IF to_regprocedure('public.search_places_semantic_v1(text,vector,integer,jsonb)') IS NULL THEN
        RAISE EXCEPTION 'search_places_semantic_v1 is missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename = 'place_embeddings_semantic_v1'
          AND indexdef ILIKE '%USING hnsw%'
    ) THEN
        RAISE EXCEPTION 'Places semantic HNSW index is missing';
    END IF;

    SELECT count(*)
    INTO embedding_count
    FROM public.place_embeddings_semantic_v1;

    IF embedding_count = 0 THEN
        RAISE EXCEPTION 'place_embeddings_semantic_v1 is empty; run the backfill first';
    END IF;

    SELECT count(*) FILTER (WHERE place.is_active)
    INTO active_count
    FROM public.place_embeddings_semantic_v1 AS place;

    IF active_count = 0 THEN
        RAISE EXCEPTION 'place_embeddings_semantic_v1 has no active rows';
    END IF;

    SELECT count(DISTINCT (
        place.embedding_model,
        place.embedding_version,
        place.metadata->>'semantic_document_version'
    ))
    INTO profile_count
    FROM public.place_embeddings_semantic_v1 AS place;

    IF profile_count <> 1 OR EXISTS (
        SELECT 1
        FROM public.place_embeddings_semantic_v1 AS place
        WHERE place.metadata->>'semantic_document_version' IS NULL
    ) THEN
        RAISE EXCEPTION 'Expected exactly one model/version/document profile, got %', profile_count;
    END IF;

    SELECT count(*)
    INTO invalid_norm_count
    FROM public.place_embeddings_semantic_v1 AS place
    WHERE abs(1.0 + (place.embedding <#> place.embedding)) > 0.02;

    IF invalid_norm_count > 0 THEN
        RAISE EXCEPTION 'Found % embeddings outside the expected unit-norm tolerance', invalid_norm_count;
    END IF;
END
$$;

-- Every provider vector is L2-normalized before it is written. The negative
-- inner product of a vector with itself is its squared norm.
SELECT
    count(*) AS rows,
    count(DISTINCT place.external_id) AS unique_ids,
    count(*) FILTER (WHERE place.is_active) AS active_rows,
    min(sqrt(GREATEST(0.0, -(place.embedding <#> place.embedding)))) AS min_norm,
    max(sqrt(GREATEST(0.0, -(place.embedding <#> place.embedding)))) AS max_norm
FROM public.place_embeddings_semantic_v1 AS place;

SELECT
    place.embedding_model,
    place.embedding_version,
    place.metadata->>'semantic_document_version' AS document_version,
    count(*) AS rows
FROM public.place_embeddings_semantic_v1 AS place
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;

-- Inspect this plan after a representative backfill. With enough rows, the
-- nearest-neighbor branch should use the HNSW index rather than materializing
-- the entire eligible corpus before ordering.
EXPLAIN (ANALYZE, BUFFERS, COSTS, VERBOSE)
SELECT place.external_id
FROM public.place_embeddings_semantic_v1 AS place
WHERE place.is_active = true
ORDER BY place.embedding <=> (
    SELECT sample.embedding
    FROM public.place_embeddings_semantic_v1 AS sample
    LIMIT 1
)
LIMIT 20;

ROLLBACK;
