from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from app.shared.nlp.preprocessing.text import normalize_text


VALID_SPLITS = {"train", "validation", "test"}
REQUIRED_FIELDS = (
    "query_id",
    "query",
    "positive_id",
    "positive",
    "negative_id",
    "negative",
    "negative_type",
    "split",
)


@dataclass(frozen=True)
class RetrievalTrainingExample:
    query_id: str
    query: str
    positive_id: str
    positive: str
    negative_id: str
    negative: str
    negative_type: str
    split: str


@dataclass(frozen=True)
class DatasetValidationReport:
    total_rows: int
    rows_by_split: dict[str, int]
    unique_queries_by_split: dict[str, int]
    negative_types: dict[str, int]


def load_retrieval_training_dataset(path: str | Path) -> list[RetrievalTrainingExample]:
    dataset_path = Path(path)
    examples: list[RetrievalTrainingExample] = []
    errors: list[str] = []

    with dataset_path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc.msg})")
                continue
            try:
                examples.append(_parse_example(payload, line_number))
            except ValueError as exc:
                errors.append(str(exc))

    errors.extend(_validate_dataset(examples))
    if errors:
        formatted = "\n- ".join(errors)
        raise ValueError(f"Invalid retrieval training dataset:\n- {formatted}")
    return examples


def dataset_validation_report(
    examples: list[RetrievalTrainingExample],
) -> DatasetValidationReport:
    rows_by_split = Counter(example.split for example in examples)
    queries_by_split: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        queries_by_split[example.split].add(example.query_id)
    return DatasetValidationReport(
        total_rows=len(examples),
        rows_by_split=dict(sorted(rows_by_split.items())),
        unique_queries_by_split={
            split: len(query_ids)
            for split, query_ids in sorted(queries_by_split.items())
        },
        negative_types=dict(
            sorted(Counter(example.negative_type for example in examples).items())
        ),
    )


def examples_for_split(
    examples: list[RetrievalTrainingExample],
    split: str,
) -> list[RetrievalTrainingExample]:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown dataset split: {split}")
    return [example for example in examples if example.split == split]


def _parse_example(payload: Any, line_number: int) -> RetrievalTrainingExample:
    if not isinstance(payload, dict):
        raise ValueError(f"line {line_number}: each JSONL row must be an object")

    values: dict[str, str] = {}
    for field in REQUIRED_FIELDS:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"line {line_number}: {field!r} must be a non-empty string"
            )
        values[field] = value.strip()
    if values["split"] not in VALID_SPLITS:
        raise ValueError(
            f"line {line_number}: split must be one of {sorted(VALID_SPLITS)}"
        )
    return RetrievalTrainingExample(**values)


def _validate_dataset(examples: list[RetrievalTrainingExample]) -> list[str]:
    errors: list[str] = []
    if not examples:
        return ["the file does not contain examples"]

    splits = {example.split for example in examples}
    for required_split in ("train", "validation"):
        if required_split not in splits:
            errors.append(f"missing required split {required_split!r}")

    query_splits: dict[str, set[str]] = defaultdict(set)
    query_texts: dict[str, set[str]] = defaultdict(set)
    document_texts: dict[str, set[str]] = defaultdict(set)
    seen_rows: set[tuple[str, ...]] = set()
    for index, example in enumerate(examples, start=1):
        query_splits[example.query_id].add(example.split)
        query_texts[example.query_id].add(normalize_text(example.query))
        document_texts[example.positive_id].add(normalize_text(example.positive))
        document_texts[example.negative_id].add(normalize_text(example.negative))

        if example.positive_id == example.negative_id:
            errors.append(f"row {index}: positive_id and negative_id must differ")
        if normalize_text(example.positive) == normalize_text(example.negative):
            errors.append(f"row {index}: positive and negative texts must differ")

        row_key = tuple(getattr(example, field) for field in REQUIRED_FIELDS)
        if row_key in seen_rows:
            errors.append(f"row {index}: exact duplicate training example")
        seen_rows.add(row_key)

    for query_id, assigned_splits in sorted(query_splits.items()):
        if len(assigned_splits) > 1:
            errors.append(
                f"query_id {query_id!r} leaks across splits: {sorted(assigned_splits)}"
            )
    for query_id, texts in sorted(query_texts.items()):
        if len(texts) > 1:
            errors.append(f"query_id {query_id!r} has inconsistent query text")
    for document_id, texts in sorted(document_texts.items()):
        if len(texts) > 1:
            errors.append(f"document id {document_id!r} has inconsistent text")
    return errors
