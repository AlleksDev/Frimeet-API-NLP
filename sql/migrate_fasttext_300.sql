-- One-time destructive migration from the old 16-dimensional mock vectors
-- to the 300-dimensional Spanish FastText vectors.
--
-- The tables contain derived search indexes. The main API remains the source
-- of truth, so old incompatible vectors are intentionally discarded.
-- Run as nlp_owner or the RDS administrator while the NLP API/jobs are paused.

BEGIN;

LOCK TABLE public.place_embeddings IN ACCESS EXCLUSIVE MODE;
LOCK TABLE public.post_embeddings IN ACCESS EXCLUSIVE MODE;

DO $$
DECLARE
    place_type TEXT;
    post_type TEXT;
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

    IF place_type <> 'vector(16)' OR post_type <> 'vector(16)' THEN
        RAISE EXCEPTION
            'Expected vector(16) columns, found place=% and post=%',
            place_type,
            post_type;
    END IF;
END $$;

DROP INDEX IF EXISTS public.place_embeddings_embedding_hnsw_idx;
DROP INDEX IF EXISTS public.post_embeddings_embedding_hnsw_idx;

TRUNCATE TABLE public.place_embeddings, public.post_embeddings;

ALTER TABLE public.place_embeddings
    ALTER COLUMN embedding TYPE VECTOR(300);

ALTER TABLE public.post_embeddings
    ALTER COLUMN embedding TYPE VECTOR(300);

CREATE INDEX place_embeddings_embedding_hnsw_idx
ON public.place_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE INDEX post_embeddings_embedding_hnsw_idx
ON public.post_embeddings USING hnsw (embedding vector_cosine_ops);

COMMIT;

-- Immediately run sql/aws_pgvector_contract.sql after this migration so the
-- match/upsert functions also declare VECTOR(300).
