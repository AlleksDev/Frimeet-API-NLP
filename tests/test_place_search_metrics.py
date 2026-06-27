import pytest

from app.modules.places.domain.search_metrics import (
    average_metrics,
    evaluate_ranking,
)


def test_evaluate_ranking_calculates_ir_metrics() -> None:
    metrics = evaluate_ranking(
        ranking=["place_a", "place_b", "place_c"],
        relevance={"place_a": 3, "place_c": 1},
        k=3,
    )

    assert metrics.precision_at_k == pytest.approx(2 / 3)
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0
    assert metrics.map == pytest.approx((1.0 + 2 / 3) / 2)
    assert 0 < metrics.ndcg_at_k <= 1


def test_average_metrics_returns_zeroes_without_queries() -> None:
    metrics = average_metrics([])

    assert metrics.precision_at_k == 0.0
    assert metrics.recall_at_k == 0.0
    assert metrics.mrr == 0.0
    assert metrics.map == 0.0
    assert metrics.ndcg_at_k == 0.0
