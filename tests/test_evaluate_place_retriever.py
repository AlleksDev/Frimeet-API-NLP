from pathlib import Path

import numpy as np

from scripts.evaluate_place_retriever import (
    build_global_corpus,
    calculate_ranks,
    read_evaluation_rows,
    summarize_ranks,
)


ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = ROOT / "data" / "training" / "places_retrieval_v1" / "test.jsonl"
VALIDATION_DATA = (
    ROOT / "data" / "training" / "places_retrieval_v1" / "validation.jsonl"
)


def test_challenge_evaluation_uses_one_global_candidate_pool() -> None:
    for path, expected_queries in ((VALIDATION_DATA, 60), (TEST_DATA, 80)):
        rows = read_evaluation_rows(path)
        corpus = build_global_corpus(rows)

        assert len(rows) == expected_queries
        assert len(corpus) == 20
        assert set(corpus) == {row["positive"] for row in rows}


def test_rank_metrics_are_calculated_over_all_documents_and_by_challenge() -> None:
    scores = np.asarray(
        [
            [0.9, 0.8, 0.1, 0.0],
            [0.8, 0.9, 0.7, 0.1],
            [0.4, 0.3, 0.2, 0.1],
        ]
    )
    ranks = calculate_ranks(scores, [0, 2, 3])
    summary = summarize_ranks(
        ranks,
        tags=[["canonical"], ["colloquial"], ["colloquial", "misspelling"]],
        documents=4,
    )

    assert ranks == [1, 3, 4]
    assert summary["queries"] == 3
    assert summary["documents"] == 4
    assert summary["top1"] == 1 / 3
    assert summary["recall_at_3"] == 2 / 3
    assert summary["per_challenge"]["colloquial"]["queries"] == 2
    assert summary["per_challenge"]["misspelling"]["mean_rank"] == 4.0
