from pathlib import Path


def test_hybrid_search_dynamic_sql_escapes_format_percent_signs() -> None:
    contract = Path("sql/aws_pgvector_contract.sql").read_text(encoding="utf-8")
    dynamic_sql = contract.split("RETURN QUERY EXECUTE format($query$", maxsplit=1)[1]
    dynamic_sql = dynamic_sql.split("$query$, target_table)", maxsplit=1)[0]

    remaining_percent_signs = dynamic_sql.replace("%%", "").replace("%I", "")

    assert "%" not in remaining_percent_signs


def test_post_feed_contract_is_synchronized() -> None:
    migration = Path("sql/migrate_post_feed_v1.sql").read_text(encoding="utf-8")
    base_contract = Path("sql/aws_pgvector_contract.sql").read_text(encoding="utf-8")
    full_setup = Path("sql/aws_pgvector_full_setup.psql.sql").read_text(encoding="utf-8")
    required = {
        "post_cluster_runs",
        "post_cluster_memberships",
        "post_embedding_tombstones",
        "user_interest_embeddings",
        "post_sync_checkpoints",
        "get_post_feed_features",
        "deactivate_post_embedding",
        "validate_post_cluster_run",
        "fail_post_cluster_run",
        "get_post_cluster_run",
        "reset_user_interest_profiles",
        "reset_post_sync_checkpoint",
        "activate_post_cluster_run",
    }
    assert all(name in migration for name in required)
    base_required = {
        "post_cluster_runs",
        "post_cluster_memberships",
        "post_embedding_tombstones",
        "user_interest_embeddings",
        "post_sync_checkpoints",
    }
    assert all(name in base_contract for name in base_required)
    assert "\\ir migrate_post_feed_v1.sql" in full_setup
    assert "\\ir migrate_post_feed_v2.sql" in full_setup
    assert Path("sql/verify_post_feed_v1.sql").exists()
    assert Path("sql/rollback_post_feed_v1.sql").exists()


def test_post_feed_v2_and_convergent_schema_are_present() -> None:
    migration = Path("sql/migrate_post_feed_v2.sql").read_text(encoding="utf-8")
    verifier = Path("sql/verify_post_feed_v2.sql").read_text(encoding="utf-8")
    convergent = Path("sql/new_pgvector_schema.sql").read_text(encoding="utf-8")

    assert "source_version es obligatorio" in migration
    assert "EXCLUDED.source_version > post_embeddings.source_version" in migration
    assert "cluster_count <> expected_k" in migration
    assert "post_embedding_dimension_ok" in verifier
    assert "aws_pgvector_contract" not in convergent  # contenido inline, sin includes
    assert "CREATE TABLE IF NOT EXISTS post_embeddings" in convergent
    assert "CREATE OR REPLACE FUNCTION public.get_post_feed_features" in convergent
    assert "migrate_post_feed_v2.sql" not in convergent  # contenido inline, sin includes


def test_global_search_candidate_filters_are_convergent_and_incremental() -> None:
    contract = Path("sql/aws_pgvector_contract.sql").read_text(encoding="utf-8")
    convergent = Path("sql/new_pgvector_schema.sql").read_text(encoding="utf-8")
    migration = Path(
        "sql/migrations/20260712_01_global_search_candidate_filters.sql"
    ).read_text(encoding="utf-8")
    verifier = Path("sql/verify_global_search_candidate_filters.sql")

    for token in (
        "event_active_at",
        "min_semantic_score",
        "min_lexical_score",
        "make_interval",
        "try_parse_timestamptz",
        "try_parse_positive_integer",
    ):
        assert token in contract
        assert token in convergent
        assert token in migration
    upper_migration = migration.upper()
    assert "BEGIN;" in upper_migration
    assert "COMMIT;" in upper_migration
    assert "CREATE OR REPLACE FUNCTION PUBLIC.SEARCH_RESOURCE_EMBEDDINGS" in upper_migration
    assert "CREATE TABLE" not in upper_migration
    assert "ALTER TABLE" not in upper_migration
    assert "CREATE INDEX" not in upper_migration
    assert "DROP FUNCTION" not in upper_migration
    assert "TO NLP_READER" in upper_migration
    assert verifier.exists()
    verifier_sql = verifier.read_text(encoding="utf-8")
    assert "RAISE EXCEPTION 'Verify search V2" in verifier_sql
    assert "SET TRANSACTION READ ONLY" in verifier_sql


def test_global_search_sql_does_not_directly_cast_untrusted_event_metadata() -> None:
    for path in (
        Path("sql/aws_pgvector_contract.sql"),
        Path("sql/new_pgvector_schema.sql"),
        Path("sql/migrations/20260712_01_global_search_candidate_filters.sql"),
    ):
        sql = path.read_text(encoding="utf-8")
        assert "NULLIF(e.metadata->>'start_time', '')::timestamptz" not in sql
        assert "(e.metadata->>'duration_minutes')::integer" not in sql


def test_places_semantic_index_is_additive_and_hybrid() -> None:
    migration = Path(
        "sql/migrations/20260716_02_places_semantic_v1.sql"
    ).read_text(encoding="utf-8")
    upper = migration.upper()

    assert "PLACE_EMBEDDINGS_SEMANTIC_V1" in upper
    assert "VECTOR(768)" in upper
    assert "SEARCH_PLACES_SEMANTIC_V1" in upper
    assert "FULL OUTER JOIN" in upper
    assert "WEBSEARCH_TO_TSQUERY" in upper
    assert "USING HNSW" in upper
    assert "AS MATERIALIZED" not in upper
    assert "FROM PUBLIC.PLACE_EMBEDDINGS_SEMANTIC_V1 AS PLACE" in upper
    assert "ALTER TABLE PUBLIC.PLACE_EMBEDDINGS" not in upper
    assert "DROP TABLE" not in upper

    verifier = Path("sql/verify_places_semantic_v1.sql")
    assert verifier.exists()
    verifier_sql = verifier.read_text(encoding="utf-8").upper()
    assert "SET TRANSACTION READ ONLY" in verifier_sql
    assert "VECTOR(768)" in verifier_sql
    assert "USING HNSW" in verifier_sql
    assert "EXPLAIN (ANALYZE, BUFFERS" in verifier_sql
