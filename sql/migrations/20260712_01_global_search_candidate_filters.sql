-- Busqueda global V2: vigencia de eventos y umbrales de relevancia.
--
-- Ejecutar UNA sola vez en la BD NLP desplegada con el propietario de la
-- funcion public.search_resource_embeddings. No crea tablas, columnas ni
-- indices; conserva la firma usada por la API y agrega dos helpers de parseo
-- seguro para que metadata temporal corrupta se descarte sin abortar la busqueda.

BEGIN;

SET LOCAL lock_timeout = '10s';
SET LOCAL statement_timeout = '10min';
SET LOCAL idle_in_transaction_session_timeout = '5min';
SET LOCAL search_path = public;

SELECT pg_advisory_xact_lock(
    hashtextextended('public:20260712_01_global_search_candidate_filters', 0)
);

DO $preflight$
DECLARE
    required_table TEXT;
    current_definition TEXT;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        RAISE EXCEPTION 'Preflight search V2: falta la extension vector';
    END IF;

    IF to_regprocedure(
        'public.search_resource_embeddings(text,text,vector,integer,jsonb)'
    ) IS NULL THEN
        RAISE EXCEPTION 'Preflight search V2: falta search_resource_embeddings(text,text,vector,integer,jsonb)';
    END IF;

    FOREACH required_table IN ARRAY ARRAY[
        'place_embeddings',
        'post_embeddings',
        'user_embeddings',
        'club_embeddings',
        'group_embeddings',
        'event_embeddings'
    ] LOOP
        IF to_regclass('public.' || required_table) IS NULL THEN
            RAISE EXCEPTION 'Preflight search V2: falta public.%', required_table;
        END IF;
    END LOOP;

    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_reader') THEN
        RAISE EXCEPTION 'Preflight search V2: falta el rol nlp_reader';
    END IF;

    SELECT pg_get_functiondef(
        'public.search_resource_embeddings(text,text,vector,integer,jsonb)'::regprocedure
    ) INTO current_definition;

    IF current_definition LIKE '%event_active_at%'
       OR current_definition LIKE '%min_semantic_score%'
       OR current_definition LIKE '%min_lexical_score%' THEN
        RAISE EXCEPTION USING
            MESSAGE = 'Preflight search V2: la funcion ya contiene cambios V2 o una instalacion parcial',
            HINT = 'No vuelva a ejecutar la migracion. Ejecute sql/verify_global_search_candidate_filters.sql.';
    END IF;
END;
$preflight$;

CREATE OR REPLACE FUNCTION public.try_parse_timestamptz(p_value TEXT)
RETURNS TIMESTAMPTZ
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
SET search_path = pg_catalog
AS $$
BEGIN
    RETURN NULLIF(btrim(p_value), '')::timestamptz;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION public.try_parse_positive_integer(p_value TEXT)
RETURNS INTEGER
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
SET search_path = pg_catalog
AS $$
DECLARE
    parsed INTEGER;
BEGIN
    parsed := NULLIF(btrim(p_value), '')::integer;
    RETURN CASE WHEN parsed > 0 THEN parsed ELSE NULL END;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION public.search_resource_embeddings(
    p_resource_type TEXT,
    p_query_text TEXT,
    p_query_embedding VECTOR(300),
    p_match_count INTEGER,
    p_filters JSONB DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    external_id TEXT,
    document TEXT,
    metadata JSONB,
    score DOUBLE PRECISION,
    semantic_score DOUBLE PRECISION,
    lexical_score DOUBLE PRECISION
)
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    target_table TEXT;
BEGIN
    target_table := CASE p_resource_type
        WHEN 'places' THEN 'place_embeddings'
        WHEN 'posts' THEN 'post_embeddings'
        WHEN 'users' THEN 'user_embeddings'
        WHEN 'clubs' THEN 'club_embeddings'
        WHEN 'groups' THEN 'group_embeddings'
        WHEN 'events' THEN 'event_embeddings'
        ELSE NULL
    END;
    IF target_table IS NULL THEN
        RAISE EXCEPTION 'Unsupported search resource type: %', p_resource_type;
    END IF;

    RETURN QUERY EXECUTE format($query$
        WITH eligible AS NOT MATERIALIZED (
            SELECT e.*
            FROM public.%I e
            WHERE e.is_active = true
              AND COALESCE(($4->>'is_active')::boolean, true) = true
              AND (
                    ($4 ? 'city') IS FALSE
                    OR lower(e.metadata->>'city') = lower($4->>'city')
              )
              AND (
                    ($4 ? 'state') IS FALSE
                    OR lower(e.metadata->>'state') = lower($4->>'state')
              )
              AND (
                    ($4 ? 'categories') IS FALSE
                    OR EXISTS (
                        SELECT 1
                        FROM jsonb_array_elements_text($4->'categories') AS allowed(value)
                        WHERE lower(e.metadata->>'category') = lower(allowed.value)
                    )
              )
              AND (
                    ($4 ? 'tags') IS FALSE
                    OR EXISTS (
                        SELECT 1
                        FROM jsonb_array_elements_text($4->'tags') AS allowed(value)
                        WHERE lower(COALESCE(e.metadata->>'tags', ''))
                              LIKE ('%%' || lower(allowed.value) || '%%')
                    )
              )
              AND (
                    ($4 ? 'published_from') IS FALSE
                    OR public.try_parse_timestamptz(e.metadata->>'published_at')
                       >= ($4->>'published_from')::timestamptz
              )
              AND (
                    ($4 ? 'published_to') IS FALSE
                    OR public.try_parse_timestamptz(e.metadata->>'published_at')
                       <= ($4->>'published_to')::timestamptz
              )
              AND (
                    ($4 ? 'event_from') IS FALSE
                    OR public.try_parse_timestamptz(e.metadata->>'start_time')
                       >= ($4->>'event_from')::timestamptz
              )
              AND (
                    ($4 ? 'event_to') IS FALSE
                    OR public.try_parse_timestamptz(e.metadata->>'start_time')
                       <= ($4->>'event_to')::timestamptz
              )
              AND (
                    ($4 ? 'event_active_at') IS FALSE
                    OR (
                        public.try_parse_timestamptz(e.metadata->>'start_time') IS NOT NULL
                        AND public.try_parse_positive_integer(
                            e.metadata->>'duration_minutes'
                        ) IS NOT NULL
                        AND public.try_parse_timestamptz(e.metadata->>'start_time')
                            + make_interval(
                                mins => public.try_parse_positive_integer(
                                    e.metadata->>'duration_minutes'
                                )
                              )
                            > ($4->>'event_active_at')::timestamptz
                    )
              )
              AND (
                    ($4 ? 'is_online') IS FALSE
                    OR COALESCE((e.metadata->>'is_online')::boolean, false)
                       = ($4->>'is_online')::boolean
              )
              AND (
                    ($4 ? 'user_roles') IS FALSE
                    OR EXISTS (
                        SELECT 1
                        FROM jsonb_array_elements_text($4->'user_roles') AS allowed(value)
                        WHERE lower(e.metadata->>'role') = lower(allowed.value)
                    )
              )
              AND (
                    ($4 ? 'place_ids') IS FALSE
                    OR e.metadata->>'place_id' IN (
                        SELECT jsonb_array_elements_text($4->'place_ids')
                    )
              )
              AND (
                    (
                        COALESCE((e.metadata->>'is_private')::boolean, false) = false
                        AND COALESCE((e.metadata->>'is_public')::boolean, true) = true
                    )
                    OR (
                        $4 ? 'requester_id'
                        AND (
                            e.metadata->>'creator_id' = $4->>'requester_id'
                            OR COALESCE(e.metadata->'authorized_user_ids', '[]'::jsonb)
                               ? ($4->>'requester_id')
                        )
                    )
              )
        ),
        vector_results AS (
            SELECT
                e.external_id,
                1 - (e.embedding <=> $1) AS semantic_score,
                row_number() OVER (
                    ORDER BY e.embedding <=> $1, e.external_id ASC
                ) AS vector_rank
            FROM eligible e
            ORDER BY e.embedding <=> $1, e.external_id ASC
            LIMIT GREATEST($3 * 3, 30)
        ),
        lexical_results AS (
            SELECT
                e.external_id,
                ts_rank_cd(e.textsearch, query.value) AS lexical_score,
                row_number() OVER (
                    ORDER BY ts_rank_cd(e.textsearch, query.value) DESC, e.external_id ASC
                ) AS lexical_rank
            FROM eligible e
            CROSS JOIN websearch_to_tsquery('simple', $2) AS query(value)
            WHERE e.textsearch @@ query.value
            ORDER BY lexical_score DESC, e.external_id ASC
            LIMIT GREATEST($3 * 3, 30)
        ),
        fused AS (
            SELECT
                COALESCE(v.external_id, l.external_id) AS external_id,
                v.semantic_score,
                l.lexical_score,
                COALESCE(1.0 / (60.0 + v.vector_rank), 0.0)
                    + COALESCE(1.0 / (60.0 + l.lexical_rank), 0.0) AS rrf_score
            FROM vector_results v
            FULL OUTER JOIN lexical_results l USING (external_id)
        )
        SELECT
            e.external_id,
            e.document,
            e.metadata,
            LEAST(
                1.0,
                f.rrf_score / (2.0 / 61.0)
                + CASE
                    WHEN lower(COALESCE(
                        e.metadata->>'search_title',
                        e.metadata->>'name',
                        e.metadata->>'title',
                        e.metadata->>'username',
                        ''
                    )) = lower($2) THEN 0.15
                    ELSE 0.0
                  END
            )::double precision AS score,
            f.semantic_score::double precision,
            f.lexical_score::double precision
        FROM fused f
        JOIN eligible e USING (external_id)
        WHERE (
            (
                ($4 ? 'min_semantic_score') IS FALSE
                AND ($4 ? 'min_lexical_score') IS FALSE
            )
            OR (
                ($4 ? 'min_semantic_score')
                AND COALESCE(
                    f.semantic_score >= ($4->>'min_semantic_score')::double precision,
                    FALSE
                )
            )
            OR (
                ($4 ? 'min_lexical_score')
                AND COALESCE(
                    f.lexical_score >= ($4->>'min_lexical_score')::double precision,
                    FALSE
                )
            )
        )
        ORDER BY score DESC, f.semantic_score DESC NULLS LAST, e.external_id ASC
        LIMIT GREATEST($3, 1)
    $query$, target_table)
    USING p_query_embedding, p_query_text, p_match_count, p_filters;
END;
$$;

REVOKE ALL ON FUNCTION public.search_resource_embeddings(
    text, text, vector, integer, jsonb
) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.try_parse_timestamptz(text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.try_parse_positive_integer(text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.search_resource_embeddings(
    text, text, vector, integer, jsonb
) TO nlp_reader;
GRANT EXECUTE ON FUNCTION public.try_parse_timestamptz(text) TO nlp_reader;
GRANT EXECUTE ON FUNCTION public.try_parse_positive_integer(text) TO nlp_reader;

DO $postcheck$
DECLARE
    installed_definition TEXT;
BEGIN
    SELECT pg_get_functiondef(
        'public.search_resource_embeddings(text,text,vector,integer,jsonb)'::regprocedure
    ) INTO installed_definition;

    IF installed_definition NOT LIKE '%event_active_at%'
       OR installed_definition NOT LIKE '%min_semantic_score%'
       OR installed_definition NOT LIKE '%min_lexical_score%'
       OR installed_definition NOT LIKE '%try_parse_timestamptz%'
       OR installed_definition NOT LIKE '%try_parse_positive_integer%' THEN
        RAISE EXCEPTION 'Postcheck search V2: la definicion instalada esta incompleta';
    END IF;

    IF NOT has_function_privilege(
        'nlp_reader',
        'public.search_resource_embeddings(text,text,vector,integer,jsonb)',
        'EXECUTE'
    ) THEN
        RAISE EXCEPTION 'Postcheck search V2: nlp_reader no conserva EXECUTE';
    END IF;

    IF NOT has_function_privilege(
        'nlp_reader', 'public.try_parse_timestamptz(text)', 'EXECUTE'
    ) OR NOT has_function_privilege(
        'nlp_reader', 'public.try_parse_positive_integer(text)', 'EXECUTE'
    ) THEN
        RAISE EXCEPTION 'Postcheck search V2: nlp_reader no puede ejecutar los helpers seguros';
    END IF;
END;
$postcheck$;

DO $functional_postcheck$
DECLARE
    zero_embedding VECTOR(300);
    invalid_count BIGINT;
BEGIN
    zero_embedding := (
        '[' || rtrim(repeat('0,', 300), ',') || ']'
    )::vector;

    SELECT COUNT(*) INTO invalid_count
    FROM public.search_resource_embeddings(
        'users',
        '__frimeet_verification_no_match_9f7c2__',
        zero_embedding,
        100,
        jsonb_build_object(
            'is_active', TRUE,
            'min_semantic_score', 1.000001,
            'min_lexical_score', 1000000.0
        )
    );
    IF invalid_count <> 0 THEN
        RAISE EXCEPTION 'Postcheck search V2: una consulta bajo umbrales imposibles devolvio % filas', invalid_count;
    END IF;

    SELECT COUNT(*) INTO invalid_count
    FROM public.search_resource_embeddings(
        'events',
        '__frimeet_event_filter_verification__',
        zero_embedding,
        1000,
        jsonb_build_object(
            'is_active', TRUE,
            'event_active_at', CURRENT_TIMESTAMP
        )
    ) result
    WHERE public.try_parse_timestamptz(result.metadata->>'start_time') IS NULL
       OR public.try_parse_positive_integer(
            result.metadata->>'duration_minutes'
          ) IS NULL
       OR public.try_parse_timestamptz(result.metadata->>'start_time')
          + make_interval(
                mins => public.try_parse_positive_integer(
                    result.metadata->>'duration_minutes'
                )
            ) <= CURRENT_TIMESTAMP;
    IF invalid_count <> 0 THEN
        RAISE EXCEPTION 'Postcheck search V2: se devolvieron % eventos terminados o invalidos', invalid_count;
    END IF;
END;
$functional_postcheck$;

COMMIT;
