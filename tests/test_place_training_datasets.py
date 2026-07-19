import json
from collections import Counter
from pathlib import Path

from scripts.build_place_training_datasets import (
    INTENT_CONCEPTS,
    _concept_key,
    _expanded_place_specs,
    _place_document,
)
from scripts.train_place_intent_bert import SLOT_TYPES, read_jsonl
from scripts.train_place_retriever import _read_training_rows


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "training"


def test_intent_dataset_is_valid_open_vocabulary_and_leak_free() -> None:
    seen: set[str] = set()
    slot_types: set[str] = set()
    train_slot_counts: Counter[str] = Counter()
    counts: dict[str, int] = {}

    for split in ("train", "validation", "test"):
        rows = read_jsonl(DATA / "places_intent_v1" / f"{split}.jsonl")
        counts[split] = len(rows)
        for row in rows:
            normalized = row.text.casefold()
            assert normalized not in seen
            seen.add(normalized)
            for span in row.spans:
                assert row.text[span.start : span.end].strip()
                slot_types.add(span.slot)
                if split == "train":
                    train_slot_counts[span.slot] += 1

    assert counts == {"train": 132, "validation": 34, "test": 34}
    assert 150 <= sum(counts.values()) <= 200
    assert slot_types == set(SLOT_TYPES)
    assert all(train_slot_counts[slot] >= 12 for slot in SLOT_TYPES)
    concept_splits = [set(INTENT_CONCEPTS[split]) for split in ("train", "validation", "test")]
    assert concept_splits[0].isdisjoint(concept_splits[1])
    assert concept_splits[0].isdisjoint(concept_splits[2])
    assert concept_splits[1].isdisjoint(concept_splits[2])
    test_text = (DATA / "places_intent_v1" / "test.jsonl").read_text(
        encoding="utf-8"
    )
    assert "observatorio astronómico" in test_text
    assert "taller de cerámica" in test_text


def test_retrieval_dataset_matches_training_contract_and_has_unique_positives() -> None:
    seen_queries: set[str] = set()
    seen_positives: set[str] = set()
    counts: dict[str, int] = {}

    for split in ("train", "validation", "test"):
        rows = _read_training_rows(
            DATA / "places_retrieval_v1" / f"{split}.jsonl"
        )
        counts[split] = len(rows)
        for row in rows:
            assert row["query"].casefold() not in seen_queries
            assert row["positive"] not in seen_positives
            seen_queries.add(row["query"].casefold())
            seen_positives.add(row["positive"])
            assert row["positive"].startswith("Nombre: ")
            assert "Tipo registrado:" in row["positive"]
            assert "Descripcion:" in row["positive"]
            assert "Etiquetas:" in row["positive"]
            assert len(row["hard_negatives"]) == 3
            assert len(set(row["hard_negatives"])) == 3
            assert row["positive"] not in row["hard_negatives"]
            assert not row["query"].startswith("query: ")
            assert not row["positive"].startswith("passage: ")

    assert counts == {"train": 160, "validation": 20, "test": 20}
    assert 150 <= sum(counts.values()) <= 200


def test_retrieval_hard_negatives_never_use_a_sibling_of_the_positive() -> None:
    for split in ("train", "validation", "test"):
        specs = _expanded_place_specs(split)
        rows = _read_training_rows(
            DATA / "places_retrieval_v1" / f"{split}.jsonl"
        )
        document_concepts = {
            _place_document(spec): _concept_key(spec) for spec in specs
        }
        assert len(rows) == len(specs)
        if split != "train":
            assert len({_concept_key(spec) for spec in specs}) == len(specs)
        for spec, row in zip(specs, rows):
            positive_concept = _concept_key(spec)
            assert document_concepts[row["positive"]] == positive_concept
            assert all(
                document_concepts[negative] != positive_concept
                for negative in row["hard_negatives"]
            )


def test_dataset_manifest_matches_generated_files() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["intent"]["train"]["records"] == 132
    assert manifest["retrieval"]["train"]["records"] == 160
