from pathlib import Path

from app.modules.places.domain.models import PlaceFilters


MIGRATION = (
    Path(__file__).parents[1]
    / "sql"
    / "migrations"
    / "20260721_03_place_facets_and_incremental_sync.sql"
)


def test_place_facet_migration_is_additive_and_idempotent() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    for required in (
        "create table if not exists public.place_sync_checkpoints",
        "create index if not exists place_semantic_v1_attribute_states_gin_idx",
        "create index if not exists place_semantic_v1_negative_attribute_terms_gin_idx",
        "place_semantic_v1_matches_facets",
        "required_attribute_states",
        "attribute_terms_any",
        "deactivate_place_embedding_semantic_v1",
        "deactivate_place_embedding",
        "get_place_sync_checkpoint",
        "save_place_sync_checkpoint",
    ):
        assert required in sql

    for destructive in ("drop table", "truncate table", "delete from public.place_embeddings"):
        assert destructive not in sql


def test_place_filters_forward_structured_facets_to_pgvector() -> None:
    filters = PlaceFilters(
        required_attribute_states={"has_parking": True},
        attribute_terms_any=("estacionamiento", "pantallas"),
        entertainment_features_any=("musica en vivo",),
    ).as_metadata_filter()

    assert filters["required_attribute_states"] == {"has_parking": True}
    assert filters["attribute_terms_any"] == ("estacionamiento", "pantallas")
    assert filters["entertainment_features_any"] == ("musica en vivo",)
