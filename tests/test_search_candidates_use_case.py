from datetime import UTC, datetime

import pytest

from app.modules.search.application.use_cases.search_candidates import (
    SearchCandidatesUseCase,
)
from app.modules.search.domain.models import SearchHit, SearchResourceType


class StubEmbeddingProvider:
    def embed_text(self, text: str) -> list[float]:
        assert text
        return [1.0, 0.0]


class StubSearchProvider:
    def __init__(
        self,
        resource_type: SearchResourceType,
        hits: list[SearchHit] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.resource_type = resource_type
        self._hits = hits or []
        self._error = error

    async def search(self, **kwargs: object) -> list[SearchHit]:
        if self._error is not None:
            raise self._error
        offset = int(kwargs["offset"])
        limit = int(kwargs["limit"])
        return self._hits[offset : offset + limit]


def _hit(identifier: str, score: float) -> SearchHit:
    return SearchHit(
        id=identifier,
        resource_type=SearchResourceType.USERS,
        title=identifier,
        score=score,
        semantic_score=score,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hits", "limit", "expected_ids", "has_more"),
    [
        ([], 2, [], False),
        ([_hit("one", 0.9)], 2, ["one"], False),
        (
            [_hit("one", 0.9), _hit("two", 0.8), _hit("three", 0.7)],
            2,
            ["one", "two"],
            True,
        ),
    ],
)
async def test_candidates_returns_zero_one_or_bounded_results_without_fill(
    hits: list[SearchHit],
    limit: int,
    expected_ids: list[str],
    has_more: bool,
) -> None:
    use_case = SearchCandidatesUseCase(
        embedding_provider=StubEmbeddingProvider(),
        providers=[StubSearchProvider(SearchResourceType.USERS, hits)],
        policy_version="test-v1",
    )

    result = await use_case.execute(
        query="consulta",
        resource_types=(SearchResourceType.USERS,),
        candidate_limit_per_type=limit,
        as_of=datetime(2026, 7, 12, tzinfo=UTC),
    )

    assert [item.id for item in result.sections[SearchResourceType.USERS]] == expected_ids
    page = result.pagination[SearchResourceType.USERS]
    assert page.returned_count == len(expected_ids)
    assert page.has_more is has_more
    assert page.next_offset == (len(expected_ids) if has_more else None)


@pytest.mark.asyncio
async def test_candidates_preserves_successful_sections_when_one_provider_fails() -> None:
    use_case = SearchCandidatesUseCase(
        embedding_provider=StubEmbeddingProvider(),
        providers=[
            StubSearchProvider(SearchResourceType.USERS, [_hit("user", 0.9)]),
            StubSearchProvider(
                SearchResourceType.EVENTS,
                error=TimeoutError("database timeout"),
            ),
        ],
        policy_version="test-v1",
    )

    result = await use_case.execute(
        query="consulta",
        resource_types=(SearchResourceType.USERS, SearchResourceType.EVENTS),
        candidate_limit_per_type=2,
        as_of=datetime(2026, 7, 12, tzinfo=UTC),
    )

    assert [item.id for item in result.sections[SearchResourceType.USERS]] == ["user"]
    assert result.sections[SearchResourceType.EVENTS] == []
    assert result.failed_resources == {SearchResourceType.EVENTS: "TimeoutError"}
    assert result.pagination[SearchResourceType.EVENTS].has_more is False
