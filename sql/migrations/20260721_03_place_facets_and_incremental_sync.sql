-- Additive contract for structured Place facets and incremental synchronization.
-- No source row or embedding is deleted, truncated, or re-embedded here.

BEGIN;

SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '5min';
SET LOCAL search_path = public;

CREATE TABLE IF NOT EXISTS public.place_sync_checkpoints (
    consumer TEXT PRIMARY KEY,
    last_event_id BIGINT NOT NULL DEFAULT 0 CHECK (last_event_id >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

DO $$
BEGIN
    IF to_regclass('public.place_embeddings_semantic_v1') IS NULL THEN
        RAISE EXCEPTION
            'place_embeddings_semantic_v1 no existe; ejecuta primero 20260716_02_places_semantic_v1.sql';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'public.place_embeddings_semantic_v1'::regclass
          AND conname = 'place_semantic_attribute_states_object'
    ) THEN
        ALTER TABLE public.place_embeddings_semantic_v1
            ADD CONSTRAINT place_semantic_attribute_states_object
            CHECK (
                (metadata ? 'attribute_states') IS FALSE
                OR jsonb_typeof(metadata->'attribute_states') = 'object'
            ) NOT VALID;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'public.place_embeddings_semantic_v1'::regclass
          AND conname = 'place_semantic_attribute_terms_array'
    ) THEN
        ALTER TABLE public.place_embeddings_semantic_v1
            ADD CONSTRAINT place_semantic_attribute_terms_array
            CHECK (
                (metadata ? 'attribute_terms') IS FALSE
                OR jsonb_typeof(metadata->'attribute_terms') = 'array'
            ) NOT VALID;
    END IF;
END
$$;

-- Expression indexes keep the authoritative payload in metadata JSONB while
-- making structured facets directly usable by SQL filters. Existing rows are
-- not rewritten and missing fields remain unknown/neutral.
CREATE INDEX IF NOT EXISTS place_semantic_v1_category_label_idx
    ON public.place_embeddings_semantic_v1 (
        lower(COALESCE(metadata->>'category_label', ''))
    );

CREATE INDEX IF NOT EXISTS place_semantic_v1_attribute_states_gin_idx
    ON public.place_embeddings_semantic_v1
    USING gin ((metadata->'attribute_states'));

CREATE INDEX IF NOT EXISTS place_semantic_v1_attribute_terms_gin_idx
    ON public.place_embeddings_semantic_v1
    USING gin ((metadata->'attribute_terms'));

CREATE INDEX IF NOT EXISTS place_semantic_v1_negative_attribute_terms_gin_idx
    ON public.place_embeddings_semantic_v1
    USING gin ((metadata->'negative_attribute_terms'));

CREATE INDEX IF NOT EXISTS place_semantic_v1_entertainment_gin_idx
    ON public.place_embeddings_semantic_v1
    USING gin ((metadata->'entertainment_features'));

CREATE INDEX IF NOT EXISTS place_semantic_v1_contained_items_gin_idx
    ON public.place_embeddings_semantic_v1
    USING gin ((metadata->'contained_items'));

CREATE INDEX IF NOT EXISTS place_semantic_v1_menu_items_gin_idx
    ON public.place_embeddings_semantic_v1
    USING gin ((metadata->'menu_items'));

CREATE OR REPLACE FUNCTION public.place_semantic_v1_matches_facets(
    place_metadata JSONB,
    filters JSONB
)
RETURNS BOOLEAN
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT
        (
            (filters ? 'required_attribute_states') IS FALSE
            OR COALESCE(place_metadata->'attribute_states', '{}'::jsonb)
               @> filters->'required_attribute_states'
        )
        AND (
            (filters ? 'attribute_terms_any') IS FALSE
            OR COALESCE(place_metadata->'attribute_terms', '[]'::jsonb)
               ?| ARRAY(
                    SELECT jsonb_array_elements_text(filters->'attribute_terms_any')
               )
        )
        AND (
            (filters ? 'entertainment_features_any') IS FALSE
            OR COALESCE(place_metadata->'entertainment_features', '[]'::jsonb)
               ?| ARRAY(
                    SELECT jsonb_array_elements_text(filters->'entertainment_features_any')
               )
        )
        AND (
            (filters ? 'contained_items_any') IS FALSE
            OR COALESCE(place_metadata->'contained_items', '[]'::jsonb)
               ?| ARRAY(
                    SELECT jsonb_array_elements_text(filters->'contained_items_any')
               )
        )
        AND (
            (filters ? 'menu_items_any') IS FALSE
            OR COALESCE(place_metadata->'menu_items', '[]'::jsonb)
               ?| ARRAY(
                    SELECT jsonb_array_elements_text(filters->'menu_items_any')
               )
        );
$$;

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
      AND public.place_semantic_v1_matches_facets(place.metadata, filters)
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
    WITH eligible AS (
        SELECT place.*
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
          AND public.place_semantic_v1_matches_facets(place.metadata, filters)
    ),
    dense AS (
        SELECT
            place.external_id,
            1 - (place.embedding <=> query_embedding) AS semantic_score,
            row_number() OVER (
                ORDER BY place.embedding <=> query_embedding, place.external_id
            ) AS dense_rank
        FROM eligible AS place
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
        FROM eligible AS place
        WHERE query_text IS NOT NULL
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
    JOIN eligible AS place USING (external_id)
    ORDER BY score DESC, place.external_id
    LIMIT GREATEST(match_count, 0);
$$;

CREATE OR REPLACE FUNCTION public.deactivate_place_embedding_semantic_v1(
    p_external_id TEXT
)
RETURNS VOID
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    UPDATE public.place_embeddings_semantic_v1
    SET is_active = FALSE,
        updated_at = now()
    WHERE external_id = p_external_id;
$$;

CREATE OR REPLACE FUNCTION public.deactivate_place_embedding(
    p_external_id TEXT
)
RETURNS VOID
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    UPDATE public.place_embeddings
    SET is_active = FALSE,
        updated_at = now()
    WHERE external_id = p_external_id;
$$;

CREATE OR REPLACE FUNCTION public.get_place_sync_checkpoint(p_consumer TEXT)
RETURNS BIGINT
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT COALESCE(
        (
            SELECT last_event_id
            FROM public.place_sync_checkpoints
            WHERE consumer = p_consumer
        ),
        0
    );
$$;

CREATE OR REPLACE FUNCTION public.save_place_sync_checkpoint(
    p_consumer TEXT,
    p_last_event_id BIGINT
)
RETURNS VOID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    IF btrim(COALESCE(p_consumer, '')) = '' OR p_last_event_id < 0 THEN
        RAISE EXCEPTION 'checkpoint de lugares invalido';
    END IF;

    INSERT INTO public.place_sync_checkpoints (consumer, last_event_id, updated_at)
    VALUES (p_consumer, p_last_event_id, now())
    ON CONFLICT (consumer) DO UPDATE SET
        last_event_id = GREATEST(
            public.place_sync_checkpoints.last_event_id,
            EXCLUDED.last_event_id
        ),
        updated_at = now();
END;
$$;

REVOKE ALL ON TABLE public.place_sync_checkpoints FROM PUBLIC;
REVOKE ALL ON FUNCTION public.place_semantic_v1_matches_facets(JSONB, JSONB) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.deactivate_place_embedding_semantic_v1(TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.deactivate_place_embedding(TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.get_place_sync_checkpoint(TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.save_place_sync_checkpoint(TEXT, BIGINT) FROM PUBLIC;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_reader') THEN
        EXECUTE 'GRANT USAGE ON SCHEMA public TO nlp_reader';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.place_semantic_v1_matches_facets(JSONB, JSONB) TO nlp_reader';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_writer') THEN
        EXECUTE 'GRANT USAGE ON SCHEMA public TO nlp_writer';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.deactivate_place_embedding_semantic_v1(TEXT) TO nlp_writer';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.deactivate_place_embedding(TEXT) TO nlp_writer';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.get_place_sync_checkpoint(TEXT) TO nlp_writer';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.save_place_sync_checkpoint(TEXT, BIGINT) TO nlp_writer';
    END IF;
END
$$;

COMMIT;
