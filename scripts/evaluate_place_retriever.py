"""Evaluate E5 place retrievers against one shared, global candidate corpus.

Unlike a per-row positive-versus-negatives check, every query is ranked against
all unique positives and hard negatives in the JSONL file.  This avoids the
four-candidate ceiling that made both the base and fine-tuned models score 1.0.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_BASE_MODEL = "intfloat/multilingual-e5-base"


def read_evaluation_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        query = _required_text(payload.get("query"), line_number, "query")
        positive = _required_text(payload.get("positive"), line_number, "positive")
        negatives = payload.get("hard_negatives", [])
        if not isinstance(negatives, list):
            raise ValueError(
                f"Line {line_number}: hard_negatives must be a list of strings"
            )
        hard_negatives = [
            _required_text(value, line_number, "hard_negatives")
            for value in negatives
        ]
        raw_tags = payload.get("challenge_tags", ["unclassified"])
        if not isinstance(raw_tags, list) or not raw_tags:
            raise ValueError(
                f"Line {line_number}: challenge_tags must be a non-empty list"
            )
        challenge_tags = [
            _required_text(value, line_number, "challenge_tags")
            for value in raw_tags
        ]
        rows.append(
            {
                "query": query,
                "positive": positive,
                "hard_negatives": hard_negatives,
                "challenge_tags": challenge_tags,
            }
        )

    if not rows:
        raise ValueError("At least one evaluation example is required")
    return rows


def build_global_corpus(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Return all candidate documents once, preserving their first-seen order."""

    corpus: list[str] = []
    seen: set[str] = set()
    materialized = list(rows)
    for row in materialized:
        candidates = (row["positive"], *row.get("hard_negatives", []))
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                corpus.append(candidate)
    return corpus


def evaluate_model(
    model: Any,
    rows: list[dict[str, Any]],
    *,
    batch_size: int = 32,
) -> dict[str, Any]:
    corpus = build_global_corpus(rows)
    document_indexes = {document: index for index, document in enumerate(corpus)}
    query_texts = [f"query: {row['query']}" for row in rows]
    passage_texts = [f"passage: {document}" for document in corpus]

    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    passage_embeddings = model.encode(
        passage_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    scores = np.asarray(query_embeddings) @ np.asarray(passage_embeddings).T
    positive_indexes = [document_indexes[row["positive"]] for row in rows]
    ranks = calculate_ranks(scores, positive_indexes)
    tags = [row["challenge_tags"] for row in rows]
    return summarize_ranks(ranks, tags=tags, documents=len(corpus))


def calculate_ranks(
    scores: np.ndarray,
    positive_indexes: list[int],
) -> list[int]:
    if scores.ndim != 2:
        raise ValueError("scores must be a two-dimensional query-document matrix")
    if scores.shape[0] != len(positive_indexes):
        raise ValueError("scores and positive_indexes must contain the same queries")

    ranks: list[int] = []
    for query_index, positive_index in enumerate(positive_indexes):
        if positive_index < 0 or positive_index >= scores.shape[1]:
            raise ValueError(f"Invalid positive index for query {query_index}")
        order = np.argsort(-scores[query_index], kind="stable")
        position = np.flatnonzero(order == positive_index)
        ranks.append(int(position[0]) + 1)
    return ranks


def summarize_ranks(
    ranks: list[int],
    *,
    tags: list[list[str]],
    documents: int,
) -> dict[str, Any]:
    if not ranks or len(ranks) != len(tags):
        raise ValueError("ranks and tags must be non-empty and have equal length")

    overall = _rank_metrics(ranks)
    tag_ranks: defaultdict[str, list[int]] = defaultdict(list)
    for rank, row_tags in zip(ranks, tags):
        for tag in set(row_tags):
            tag_ranks[tag].append(rank)

    return {
        "queries": len(ranks),
        "documents": documents,
        **overall,
        "per_challenge": {
            tag: {"queries": len(values), **_rank_metrics(values)}
            for tag, values in sorted(tag_ranks.items())
        },
        "ranks": ranks,
    }


def _rank_metrics(ranks: list[int]) -> dict[str, float]:
    values = np.asarray(ranks, dtype=np.float64)
    discounted_gains_at_10 = np.where(
        values <= 10,
        1.0 / np.log2(values + 1.0),
        0.0,
    )
    return {
        "top1": float(np.mean(values <= 1)),
        "recall_at_3": float(np.mean(values <= 3)),
        "recall_at_5": float(np.mean(values <= 5)),
        "recall_at_10": float(np.mean(values <= 10)),
        "mrr": float(np.mean(1.0 / values)),
        "ndcg_at_10": float(np.mean(discounted_gains_at_10)),
        "mean_rank": float(np.mean(values)),
    }


def _required_text(value: Any, line_number: int, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line_number}: {field} must be a non-empty string")
    return " ".join(value.split())


def _parse_model_specs(values: list[str]) -> list[tuple[str, str]]:
    if not values:
        return [("base", DEFAULT_BASE_MODEL)]
    parsed: list[tuple[str, str]] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError("Each --model must use LABEL=MODEL_OR_PATH")
        label, model_path = (part.strip() for part in value.split("=", 1))
        if not label or not model_path:
            raise ValueError("Each --model must use LABEL=MODEL_OR_PATH")
        if label in labels:
            raise ValueError(f"Duplicate model label: {label}")
        labels.add(label)
        parsed.append((label, model_path))
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", required=True)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="LABEL=MODEL_OR_PATH",
        help="Repeat to compare multiple models on exactly the same corpus.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be greater than zero")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-training.txt before evaluating retrievers"
        ) from exc

    rows = read_evaluation_rows(Path(args.test_file))
    results: dict[str, Any] = {}
    for label, model_path in _parse_model_specs(args.model):
        print(f"Evaluating {label}: {model_path}")
        model = SentenceTransformer(model_path, device=args.device)
        results[label] = evaluate_model(
            model,
            rows,
            batch_size=args.batch_size,
        )

    output = json.dumps(results, ensure_ascii=False, indent=2)
    print(output)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
