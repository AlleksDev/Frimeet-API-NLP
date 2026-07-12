from datetime import UTC, datetime

from app.modules.search.domain.filters import SearchCriteria, filter_and_rank_hits
from app.modules.search.domain.models import SearchHit, SearchResourceType
from app.modules.search.domain.relevance import SearchRelevanceThreshold


def _hit(
    semantic_score: float | None,
    lexical_score: float | None,
) -> SearchHit:
    return SearchHit(
        id="candidate",
        resource_type=SearchResourceType.POSTS,
        title="candidate",
        score=0.9,
        semantic_score=semantic_score,
        lexical_score=lexical_score,
    )


def test_relevance_threshold_accepts_either_absolute_signal() -> None:
    threshold = SearchRelevanceThreshold(semantic_min=0.40, lexical_min=0.10)

    assert threshold.accepts(_hit(0.40, None))
    assert threshold.accepts(_hit(0.20, 0.10))
    assert not threshold.accepts(_hit(0.399, 0.099))
    assert not threshold.accepts(_hit(None, None))


def test_event_active_filter_keeps_in_progress_and_future_only() -> None:
    active_at = datetime(2026, 7, 12, 12, 0, tzinfo=UTC)
    criteria = SearchCriteria(event_active_at=active_at)
    hits = [
        SearchHit(
            id="ended",
            resource_type=SearchResourceType.EVENTS,
            title="ended",
            score=0.9,
            semantic_score=0.9,
            metadata={
                "start_time": "2026-07-12T10:00:00Z",
                "duration_minutes": 60,
            },
        ),
        SearchHit(
            id="in-progress",
            resource_type=SearchResourceType.EVENTS,
            title="in-progress",
            score=0.8,
            semantic_score=0.8,
            metadata={
                "start_time": "2026-07-12T11:30:00Z",
                "duration_minutes": 90,
            },
        ),
        SearchHit(
            id="future",
            resource_type=SearchResourceType.EVENTS,
            title="future",
            score=0.7,
            semantic_score=0.7,
            metadata={
                "start_time": "2026-07-13T10:00:00Z",
                "duration_minutes": 60,
            },
        ),
        SearchHit(
            id="invalid",
            resource_type=SearchResourceType.EVENTS,
            title="invalid",
            score=1.0,
            semantic_score=1.0,
            metadata={"start_time": "invalid", "duration_minutes": 60},
        ),
    ]

    ranked = filter_and_rank_hits(SearchResourceType.EVENTS, hits, criteria)

    assert [item.id for item in ranked] == ["in-progress", "future"]
