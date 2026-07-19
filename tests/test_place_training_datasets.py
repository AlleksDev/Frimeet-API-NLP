import json
from collections import Counter
from pathlib import Path

from scripts.build_place_training_datasets import (
    INTENT_CONCEPTS,
    RETRIEVAL_TEST_QUERY_VARIANTS,
    RETRIEVAL_VALIDATION_QUERY_VARIANTS,
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
    positive_splits: dict[str, str] = {}
    counts: dict[str, int] = {}

    for split in ("train", "validation", "test"):
        rows = _read_training_rows(
            DATA / "places_retrieval_v1" / f"{split}.jsonl"
        )
        counts[split] = len(rows)
        for row in rows:
            assert row["query"].casefold() not in seen_queries
            positive_owner = positive_splits.get(row["positive"])
            assert positive_owner in (None, split)
            seen_queries.add(row["query"].casefold())
            positive_splits[row["positive"]] = split
            assert row["positive"].startswith("Nombre: ")
            assert "Tipo registrado:" in row["positive"]
            assert "Descripcion:" in row["positive"]
            assert "Etiquetas:" in row["positive"]
            expected_negatives = 3 if split == "train" else 7
            assert len(row["hard_negatives"]) == expected_negatives
            assert len(set(row["hard_negatives"])) == expected_negatives
            assert row["positive"] not in row["hard_negatives"]
            assert not row["query"].startswith("query: ")
            assert not row["positive"].startswith("passage: ")

    assert counts == {"train": 160, "validation": 60, "test": 80}
    assert 150 <= counts["train"] <= 200
    assert len(positive_splits) == 200


def test_retrieval_hard_negatives_never_use_a_sibling_of_the_positive() -> None:
    for split in ("train", "validation", "test"):
        specs = _expanded_place_specs(split)
        rows = _read_training_rows(
            DATA / "places_retrieval_v1" / f"{split}.jsonl"
        )
        document_concepts = {
            _place_document(spec): _concept_key(spec) for spec in specs
        }
        if split == "train":
            assert len(rows) == len(specs)
        if split != "train":
            assert len({_concept_key(spec) for spec in specs}) == len(specs)
        assert {row["positive"] for row in rows} == set(document_concepts)
        for row in rows:
            positive_concept = document_concepts[row["positive"]]
            assert all(
                document_concepts[negative] != positive_concept
                for negative in row["hard_negatives"]
            )


def test_retrieval_evaluation_has_curated_challenges_and_repeated_qrels() -> None:
    assert len(RETRIEVAL_VALIDATION_QUERY_VARIANTS) == 20
    assert len(RETRIEVAL_TEST_QUERY_VARIANTS) == 20
    for split, expected_rows, expected_repetitions in (
        ("validation", 60, 3),
        ("test", 80, 4),
    ):
        path = DATA / "places_retrieval_v1" / f"{split}.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        positive_counts = Counter(row["positive"] for row in rows)
        tag_counts = Counter(
            tag
            for row in rows
            for tag in row["challenge_tags"]
        )

        assert len(rows) == expected_rows
        assert set(positive_counts.values()) == {expected_repetitions}
        assert tag_counts["canonical"] == 20
        assert tag_counts["colloquial"] >= 10
        assert tag_counts["implicit"] >= 20

    test_rows = [
        json.loads(line)
        for line in (DATA / "places_retrieval_v1" / "test.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    test_tag_counts = Counter(
        tag for row in test_rows for tag in row["challenge_tags"]
    )
    assert test_tag_counts["misspelling"] >= 10
    assert test_tag_counts["contrastive"] >= 4


def test_dataset_manifest_matches_generated_files() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["intent"]["train"]["records"] == 132
    assert manifest["retrieval"]["train"]["records"] == 160
    assert manifest["retrieval"]["validation"]["records"] == 60
    assert manifest["retrieval"]["test"]["records"] == 80
