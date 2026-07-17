BEGIN;
SET TRANSACTION READ ONLY;

DO $$
DECLARE
    embedding_type TEXT;
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
END
$$;

-- Inspect this plan after a representative backfill. With enough rows, the
-- nearest-neighbor branch should use the HNSW index rather than materializing
-- the entire eligible corpus before ordering.
EXPLAIN (ANALYZE, BUFFERS, COSTS, VERBOSE)
SELECT place.external_id
FROM public.place_embeddings_semantic_v1 AS place
WHERE place.is_active = true
ORDER BY place.embedding <=> (array_fill(0::real, ARRAY[768])::vector)
LIMIT 20;

ROLLBACK;
