from pathlib import Path

import pytest

from app.shared.nlp.embeddings.training_dataset import (
    dataset_validation_report,
    load_retrieval_training_dataset,
)


def test_example_retrieval_dataset_is_valid() -> None:
    dataset = Path("examples/training/place_retrieval.example.jsonl")

    examples = load_retrieval_training_dataset(dataset)
    report = dataset_validation_report(examples)

    assert report.total_rows == 8
    assert report.unique_queries_by_split == {"train": 5, "validation": 2}
    assert report.negative_types["spatial_reference_wrong_type"] == 1


def test_retrieval_dataset_rejects_query_leakage(tmp_path: Path) -> None:
    dataset = tmp_path / "leaky.jsonl"
    dataset.write_text(
        "\n".join(
            [
                '{"query_id":"q1","query":"café","positive_id":"p1",'
                '"positive":"cafetería","negative_id":"n1","negative":"parque",'
                '"negative_type":"wrong_type","split":"train"}',
                '{"query_id":"q1","query":"café","positive_id":"p1",'
                '"positive":"cafetería","negative_id":"n2","negative":"museo",'
                '"negative_type":"wrong_type","split":"validation"}',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="leaks across splits"):
        load_retrieval_training_dataset(dataset)
