-- Additive Places-only Sentence-Transformer index (multilingual-e5-base: 768d).
-- This migration never alters or drops the existing FastText VECTOR(300) table.

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.place_embeddings_semantic_v1 (
    external_id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(768) NOT NULL,
    content_hash TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT true,
    textsearch TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('simple', document)
    ) STORED,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS place_embeddings_semantic_v1_embedding_hnsw_idx
ON public.place_embeddings_semantic_v1
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS place_embeddings_semantic_v1_textsearch_gin_idx
ON public.place_embeddings_semantic_v1 USING gin (textsearch);

CREATE INDEX IF NOT EXISTS place_embeddings_semantic_v1_metadata_gin_idx
ON public.place_embeddings_semantic_v1 USING gin (metadata);

CREATE OR REPLACE FUNCTION public.match_places_semantic_v1(
    query_embedding VECTOR(768),
    match_count INTEGER,
    filters JSONB DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    external_id TEXT,
    document TEXT,
    metadata JSONB,
    score DOUBLE PRECISION
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT
        place.external_id,
        place.document,
        place.metadata,
        1 - (place.embedding <=> query_embedding) AS score
    FROM public.place_embeddings_semantic_v1 AS place
    WHERE place.is_active = true
      AND COALESCE((filters->>'is_active')::boolean, true) = true
      AND ((filters ? 'city') IS FALSE OR lower(place.metadata->>'city') = lower(filters->>'city'))
      AND ((filters ? 'state') IS FALSE OR lower(place.metadata->>'state') = lower(filters->>'state'))
      AND ((filters ? 'category') IS FALSE OR lower(place.metadata->>'category') = lower(filters->>'category'))
      AND ((filters ? 'price_range') IS FALSE OR place.metadata->>'price_range' = filters->>'price_range')
      AND ((filters ? 'occasion') IS FALSE OR place.metadata->>'occasion' ILIKE ('%' || (filters->>'occasion') || '%'))
      AND (
          (filters ? 'place_ids') IS FALSE
          OR place.external_id IN (SELECT jsonb_array_elements_text(filters->'place_ids'))
      )
    ORDER BY place.embedding <=> query_embedding, place.external_id
    LIMIT GREATEST(match_count, 0);
$$;

CREATE OR REPLACE FUNCTION public.search_places_semantic_v1(
    query_text TEXT,
    query_embedding VECTOR(768),
    match_count INTEGER,
    filters JSONB DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    external_id TEXT,
    document TEXT,
    metadata JSONB,
    score DOUBLE PRECISION,
    semantic_score DOUBLE PRECISION,
    lexical_score DOUBLE PRECISION
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    WITH dense AS (
        SELECT
            place.external_id,
            1 - (place.embedding <=> query_embedding) AS semantic_score,
            row_number() OVER (
                ORDER BY place.embedding <=> query_embedding, place.external_id
            ) AS dense_rank
        FROM public.place_embeddings_semantic_v1 AS place
        WHERE place.is_active = true
          AND COALESCE((filters->>'is_active')::boolean, true) = true
          AND ((filters ? 'city') IS FALSE OR lower(place.metadata->>'city') = lower(filters->>'city'))
          AND ((filters ? 'state') IS FALSE OR lower(place.metadata->>'state') = lower(filters->>'state'))
          AND ((filters ? 'category') IS FALSE OR lower(place.metadata->>'category') = lower(filters->>'category'))
          AND ((filters ? 'price_range') IS FALSE OR place.metadata->>'price_range' = filters->>'price_range')
          AND ((filters ? 'occasion') IS FALSE OR place.metadata->>'occasion' ILIKE ('%' || (filters->>'occasion') || '%'))
          AND (
              (filters ? 'place_ids') IS FALSE
              OR place.external_id IN (SELECT jsonb_array_elements_text(filters->'place_ids'))
          )
        ORDER BY place.embedding <=> query_embedding, place.external_id
        LIMIT GREATEST(match_count * 4, 100)
    ),
    lexical AS (
        SELECT
            place.external_id,
            ts_rank_cd(
                place.textsearch,
                websearch_to_tsquery('simple', COALESCE(query_text, ''))
            )::DOUBLE PRECISION AS lexical_score,
            row_number() OVER (
                ORDER BY
                    ts_rank_cd(
                        place.textsearch,
                        websearch_to_tsquery('simple', COALESCE(query_text, ''))
                    ) DESC,
                    place.external_id
            ) AS lexical_rank
        FROM public.place_embeddings_semantic_v1 AS place
        WHERE place.is_active = true
          AND COALESCE((filters->>'is_active')::boolean, true) = true
          AND ((filters ? 'city') IS FALSE OR lower(place.metadata->>'city') = lower(filters->>'city'))
          AND ((filters ? 'state') IS FALSE OR lower(place.metadata->>'state') = lower(filters->>'state'))
          AND ((filters ? 'category') IS FALSE OR lower(place.metadata->>'category') = lower(filters->>'category'))
          AND ((filters ? 'price_range') IS FALSE OR place.metadata->>'price_range' = filters->>'price_range')
          AND ((filters ? 'occasion') IS FALSE OR place.metadata->>'occasion' ILIKE ('%' || (filters->>'occasion') || '%'))
          AND (
              (filters ? 'place_ids') IS FALSE
              OR place.external_id IN (SELECT jsonb_array_elements_text(filters->'place_ids'))
          )
          AND query_text IS NOT NULL
          AND btrim(query_text) <> ''
          AND place.textsearch @@ websearch_to_tsquery('simple', query_text)
        ORDER BY lexical_score DESC, place.external_id
        LIMIT GREATEST(match_count * 4, 100)
    ),
    unioned AS (
        SELECT
            COALESCE(dense.external_id, lexical.external_id) AS external_id,
            dense.semantic_score,
            lexical.lexical_score,
            dense.dense_rank,
            lexical.lexical_rank
        FROM dense
        FULL OUTER JOIN lexical USING (external_id)
    ),
    scored AS (
        SELECT
            unioned.*,
            (
                COALESCE(1.0 / (60.0 + unioned.dense_rank), 0.0)
                + COALESCE(1.0 / (60.0 + unioned.lexical_rank), 0.0)
            ) AS rrf_score
        FROM unioned
    )
    SELECT
        place.external_id,
        place.document,
        place.metadata,
        LEAST(
            1.0,
            0.75 * GREATEST(COALESCE(scored.semantic_score, 0.0), 0.0)
            + 0.15 * LEAST(COALESCE(scored.lexical_score, 0.0), 1.0)
            + 0.10 * LEAST(scored.rrf_score * 31.0, 1.0)
        ) AS score,
        scored.semantic_score,
        scored.lexical_score
    FROM scored
    JOIN public.place_embeddings_semantic_v1 AS place USING (external_id)
    ORDER BY score DESC, place.external_id
    LIMIT GREATEST(match_count, 0);
$$;

CREATE OR REPLACE FUNCTION public.upsert_place_embedding_semantic_v1(
    p_external_id TEXT,
    p_document TEXT,
    p_metadata JSONB,
    p_embedding VECTOR(768),
    p_content_hash TEXT,
    p_embedding_model TEXT,
    p_embedding_version TEXT,
    p_is_active BOOLEAN
)
RETURNS VOID
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    INSERT INTO public.place_embeddings_semantic_v1 (
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

CREATE OR REPLACE FUNCTION public.get_place_content_hashes_semantic_v1(
    p_external_ids TEXT[]
)
RETURNS TABLE (external_id TEXT, content_hash TEXT)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT place.external_id, place.content_hash
    FROM public.place_embeddings_semantic_v1 AS place
    WHERE place.external_id = ANY(p_external_ids);
$$;

REVOKE ALL ON TABLE public.place_embeddings_semantic_v1 FROM PUBLIC;
REVOKE ALL ON FUNCTION public.match_places_semantic_v1(VECTOR, INTEGER, JSONB) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.search_places_semantic_v1(TEXT, VECTOR, INTEGER, JSONB) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.upsert_place_embedding_semantic_v1(TEXT, TEXT, JSONB, VECTOR, TEXT, TEXT, TEXT, BOOLEAN) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_place_content_hashes_semantic_v1(TEXT[]) FROM PUBLIC;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_reader') THEN
        EXECUTE 'GRANT USAGE ON SCHEMA public TO nlp_reader';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.match_places_semantic_v1(VECTOR, INTEGER, JSONB) TO nlp_reader';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.search_places_semantic_v1(TEXT, VECTOR, INTEGER, JSONB) TO nlp_reader';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_writer') THEN
        EXECUTE 'GRANT USAGE ON SCHEMA public TO nlp_writer';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.upsert_place_embedding_semantic_v1(TEXT, TEXT, JSONB, VECTOR, TEXT, TEXT, TEXT, BOOLEAN) TO nlp_writer';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.get_place_content_hashes_semantic_v1(TEXT[]) TO nlp_writer';
    END IF;
END
$$;

-- Execute after creating the least-privilege roles used by the service:
-- GRANT EXECUTE ON FUNCTION public.match_places_semantic_v1(VECTOR, INTEGER, JSONB) TO nlp_reader;
-- GRANT EXECUTE ON FUNCTION public.search_places_semantic_v1(TEXT, VECTOR, INTEGER, JSONB) TO nlp_reader;
-- GRANT EXECUTE ON FUNCTION public.upsert_place_embedding_semantic_v1(TEXT, TEXT, JSONB, VECTOR, TEXT, TEXT, TEXT, BOOLEAN) TO nlp_writer;
-- GRANT EXECUTE ON FUNCTION public.get_place_content_hashes_semantic_v1(TEXT[]) TO nlp_writer;

COMMIT;
