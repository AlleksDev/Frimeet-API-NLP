-- Verificador de solo lectura para Busqueda Global V2.
-- Ejecutar despues de sql/migrations/20260712_01_global_search_candidate_filters.sql.

BEGIN;
SET TRANSACTION READ ONLY;

WITH function_contract AS (
    SELECT
        to_regprocedure(
            'public.search_resource_embeddings(text,text,vector,integer,jsonb)'
        ) AS function_oid
), definition AS (
    SELECT pg_get_functiondef(function_oid) AS body
    FROM function_contract
    WHERE function_oid IS NOT NULL
)
SELECT
    (SELECT function_oid IS NOT NULL FROM function_contract) AS function_exists,
    COALESCE((SELECT body LIKE '%event_active_at%' FROM definition), FALSE)
        AS has_event_active_filter,
    COALESCE((SELECT body LIKE '%min_semantic_score%' FROM definition), FALSE)
        AS has_semantic_threshold,
    COALESCE((SELECT body LIKE '%min_lexical_score%' FROM definition), FALSE)
        AS has_lexical_threshold,
    COALESCE((SELECT body LIKE '%try_parse_timestamptz%' FROM definition), FALSE)
        AS has_safe_timestamp_parser,
    to_regprocedure('public.try_parse_positive_integer(text)') IS NOT NULL
        AS has_safe_duration_parser,
    has_function_privilege(
        'nlp_reader',
        'public.search_resource_embeddings(text,text,vector,integer,jsonb)',
        'EXECUTE'
    ) AS reader_can_execute;

WITH zero_embedding AS (
    SELECT (
        '[' || rtrim(repeat('0,', 300), ',') || ']'
    )::vector AS value
), negative_results AS (
    SELECT result.external_id
    FROM zero_embedding embedding
    CROSS JOIN LATERAL public.search_resource_embeddings(
        'users',
        '__frimeet_verification_no_match_9f7c2__',
        embedding.value,
        100,
        jsonb_build_object(
            'is_active', TRUE,
            'min_semantic_score', 1.000001,
            'min_lexical_score', 1000000.0
        )
    ) result
)
SELECT COUNT(*) AS negative_query_results
FROM negative_results;

WITH zero_embedding AS (
    SELECT (
        '[' || rtrim(repeat('0,', 300), ',') || ']'
    )::vector AS value
), event_results AS (
    SELECT result.metadata
    FROM zero_embedding embedding
    CROSS JOIN LATERAL public.search_resource_embeddings(
        'events',
        '__frimeet_event_filter_verification__',
        embedding.value,
        1000,
        jsonb_build_object(
            'is_active', TRUE,
            'event_active_at', CURRENT_TIMESTAMP
        )
    ) result
)
SELECT COUNT(*) AS ended_or_invalid_events_returned
FROM event_results
WHERE public.try_parse_timestamptz(metadata->>'start_time') IS NULL
   OR public.try_parse_positive_integer(metadata->>'duration_minutes') IS NULL
   OR public.try_parse_timestamptz(metadata->>'start_time')
      + make_interval(
            mins => public.try_parse_positive_integer(
                metadata->>'duration_minutes'
            )
        )
      <= CURRENT_TIMESTAMP;

DO $assertions$
DECLARE
    function_body TEXT;
    zero_embedding VECTOR(300);
    invalid_count BIGINT;
BEGIN
    IF to_regprocedure(
        'public.search_resource_embeddings(text,text,vector,integer,jsonb)'
    ) IS NULL THEN
        RAISE EXCEPTION 'Verify search V2: falta search_resource_embeddings';
    END IF;
    IF to_regprocedure('public.try_parse_timestamptz(text)') IS NULL
       OR to_regprocedure('public.try_parse_positive_integer(text)') IS NULL THEN
        RAISE EXCEPTION 'Verify search V2: faltan helpers de parseo seguro';
    END IF;

    SELECT pg_get_functiondef(
        'public.search_resource_embeddings(text,text,vector,integer,jsonb)'::regprocedure
    ) INTO function_body;
    IF function_body NOT LIKE '%event_active_at%'
       OR function_body NOT LIKE '%min_semantic_score%'
       OR function_body NOT LIKE '%min_lexical_score%'
       OR function_body NOT LIKE '%try_parse_timestamptz%'
       OR function_body NOT LIKE '%try_parse_positive_integer%' THEN
        RAISE EXCEPTION 'Verify search V2: contrato funcional incompleto';
    END IF;

    IF NOT has_function_privilege(
        'nlp_reader',
        'public.search_resource_embeddings(text,text,vector,integer,jsonb)',
        'EXECUTE'
    ) THEN
        RAISE EXCEPTION 'Verify search V2: nlp_reader no puede ejecutar la busqueda';
    END IF;

    SELECT COUNT(*) INTO invalid_count
    FROM public.event_embeddings
    WHERE is_active = TRUE
      AND (
            public.try_parse_timestamptz(metadata->>'start_time') IS NULL
            OR public.try_parse_positive_integer(
                metadata->>'duration_minutes'
            ) IS NULL
      );
    IF invalid_count <> 0 THEN
        RAISE EXCEPTION USING
            MESSAGE = format(
                'Verify search V2: existen %s eventos activos con metadata temporal invalida',
                invalid_count
            ),
            HINT = 'Ejecute nuevamente sync_search_embeddings para events antes de activar search.';
    END IF;

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
        RAISE EXCEPTION 'Verify search V2: consulta negativa devolvio % filas', invalid_count;
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
        RAISE EXCEPTION 'Verify search V2: se devolvieron % eventos invalidos', invalid_count;
    END IF;
END;
$assertions$;

COMMIT;
