-- Ejecutar completo una sola vez desde pgAdmin > Query Tool.
-- Migra los indices derivados FastText VECTOR(300) a E5/SBERT VECTOR(384).
-- Pausar antes el Space y los jobs, y crear un snapshot reciente de RDS.

CREATE EXTENSION IF NOT EXISTS vector;

BEGIN;

LOCK TABLE public.place_embeddings IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.post_embeddings IN ACCESS EXCLUSIVE MODE;

DO $$
DECLARE
    place_type text;
    post_type text;
BEGIN
    SELECT format_type(attribute.atttypid, attribute.atttypmod)
    INTO place_type
    FROM pg_attribute attribute
    WHERE attribute.attrelid = 'public.place_embeddings'::regclass
      AND attribute.attname = 'embedding'
      AND NOT attribute.attisdropped;

    SELECT format_type(attribute.atttypid, attribute.atttypmod)
    INTO post_type
    FROM pg_attribute attribute
    WHERE attribute.attrelid = 'public.post_embeddings'::regclass
      AND attribute.attname = 'embedding'
      AND NOT attribute.attisdropped;

    IF place_type <> 'vector(300)' OR post_type <> 'vector(300)' THEN
        RAISE EXCEPTION
            'Se esperaba VECTOR(300). Tipos encontrados: places=%, posts=%',
            place_type,
            post_type;
    END IF;
END $$;

DROP FUNCTION IF EXISTS public.match_places(vector, integer, jsonb);
DROP FUNCTION IF EXISTS public.match_posts(vector, integer, jsonb);
DROP FUNCTION IF EXISTS public.upsert_place_embedding(
    text, text, jsonb, vector, text, text, text, boolean
);
DROP FUNCTION IF EXISTS public.upsert_post_embedding(
    text, text, jsonb, vector, text, text, text, boolean
);

DROP INDEX IF EXISTS public.place_embeddings_embedding_hnsw_idx;
DROP INDEX IF EXISTS public.post_embeddings_embedding_hnsw_idx;

-- No existe una conversión matemática válida entre espacios de modelos
-- diferentes. Estas tablas son índices derivados; la API principal conserva
-- los datos fuente y los jobs las reconstruyen después de esta transacción.
TRUNCATE TABLE public.place_embeddings, public.post_embeddings;

ALTER TABLE public.place_embeddings
    ALTER COLUMN embedding TYPE vector(384);

ALTER TABLE public.post_embeddings
    ALTER COLUMN embedding TYPE vector(384);

CREATE INDEX place_embeddings_embedding_hnsw_idx
ON public.place_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX post_embeddings_embedding_hnsw_idx
ON public.post_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS place_embeddings_metadata_gin_idx
ON public.place_embeddings USING gin (metadata);

CREATE INDEX IF NOT EXISTS post_embeddings_metadata_gin_idx
ON public.post_embeddings USING gin (metadata);

CREATE OR REPLACE FUNCTION public.match_places(
    query_embedding vector(384),
    match_count integer,
    filters jsonb DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    external_id text,
    document text,
    metadata jsonb,
    score double precision
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
    FROM public.place_embeddings place
    WHERE place.is_active = true
      AND COALESCE((filters->>'is_active')::boolean, true) = true
      AND ((filters ? 'city') IS FALSE OR lower(place.metadata->>'city') = lower(filters->>'city'))
      AND ((filters ? 'state') IS FALSE OR lower(place.metadata->>'state') = lower(filters->>'state'))
      AND ((filters ? 'category') IS FALSE OR lower(place.metadata->>'category') = lower(filters->>'category'))
      AND ((filters ? 'price_range') IS FALSE OR place.metadata->>'price_range' = filters->>'price_range')
      AND ((filters ? 'occasion') IS FALSE OR place.metadata->>'occasion' ILIKE ('%' || (filters->>'occasion') || '%'))
      AND (
          (filters ? 'place_ids') IS FALSE
          OR place.external_id IN (
              SELECT jsonb_array_elements_text(filters->'place_ids')
          )
      )
    ORDER BY place.embedding <=> query_embedding
    LIMIT match_count;
$$;

CREATE OR REPLACE FUNCTION public.match_posts(
    query_embedding vector(384),
    match_count integer,
    filters jsonb DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    external_id text,
    document text,
    metadata jsonb,
    score double precision
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT
        post.external_id,
        post.document,
        post.metadata,
        1 - (post.embedding <=> query_embedding) AS score
    FROM public.post_embeddings post
    WHERE post.is_active = true
      AND COALESCE((filters->>'is_active')::boolean, true) = true
      AND ((filters ? 'city') IS FALSE OR lower(post.metadata->>'city') = lower(filters->>'city'))
    ORDER BY post.embedding <=> query_embedding
    LIMIT match_count;
$$;

CREATE OR REPLACE FUNCTION public.upsert_place_embedding(
    p_external_id text,
    p_document text,
    p_metadata jsonb,
    p_embedding vector(384),
    p_content_hash text,
    p_embedding_model text,
    p_embedding_version text,
    p_is_active boolean
)
RETURNS void
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    INSERT INTO public.place_embeddings (
        external_id, document, metadata, embedding, content_hash,
        embedding_model, embedding_version, is_active, updated_at
    )
    VALUES (
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

CREATE OR REPLACE FUNCTION public.upsert_post_embedding(
    p_external_id text,
    p_document text,
    p_metadata jsonb,
    p_embedding vector(384),
    p_content_hash text,
    p_embedding_model text,
    p_embedding_version text,
    p_is_active boolean
)
RETURNS void
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
    INSERT INTO public.post_embeddings (
        external_id, document, metadata, embedding, content_hash,
        embedding_model, embedding_version, is_active, updated_at
    )
    VALUES (
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

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_reader') THEN
        EXECUTE 'GRANT USAGE ON SCHEMA public TO nlp_reader';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.match_places(vector, integer, jsonb) TO nlp_reader';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.match_posts(vector, integer, jsonb) TO nlp_reader';
    END IF;

    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'nlp_writer') THEN
        EXECUTE 'GRANT USAGE ON SCHEMA public TO nlp_writer';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.get_place_content_hashes(text[]) TO nlp_writer';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.get_post_content_hashes(text[]) TO nlp_writer';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.upsert_place_embedding(text, text, jsonb, vector, text, text, text, boolean) TO nlp_writer';
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.upsert_post_embedding(text, text, jsonb, vector, text, text, text, boolean) TO nlp_writer';
    END IF;
END $$;

COMMIT;

SELECT
    column_info.table_name,
    format_type(attribute.atttypid, attribute.atttypmod) AS formatted_type
FROM information_schema.columns column_info
JOIN pg_attribute attribute
  ON attribute.attrelid = (
      quote_ident(column_info.table_schema) || '.' || quote_ident(column_info.table_name)
  )::regclass
 AND attribute.attname = column_info.column_name
WHERE column_info.table_schema = 'public'
  AND column_info.table_name IN ('place_embeddings', 'post_embeddings')
  AND column_info.column_name = 'embedding'
ORDER BY column_info.table_name;

SELECT 'place_embeddings' AS table_name, count(*) AS rows
FROM public.place_embeddings
UNION ALL
SELECT 'post_embeddings' AS table_name, count(*) AS rows
FROM public.post_embeddings;
