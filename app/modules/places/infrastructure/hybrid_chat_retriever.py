import json
import math
from typing import Sequence

from app.modules.places.application.ports.place_repository import PlaceVectorRepository
from app.modules.places.domain.chat_intent import (
    ParsedPlaceChatIntent,
    PlaceChatCandidate,
)
from app.modules.places.domain.chat_ranking import (
    ContentRankingWeights,
    normalize_bm25,
)
from app.modules.places.domain.models import PlaceCandidate, PlaceFilters
from app.modules.places.domain.search_text import place_tokens, tokenize
from app.modules.places.infrastructure.bm25_place_ranker import bm25_score, idf_bm25
from app.shared.cache.memory import SimpleTTLCache
from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.preprocessing.text import prepare_for_embedding


_EXACT_ENTITY_PREFERENCES = {"hello_kitty"}


class HybridContentPlaceChatRetriever:
    """Fuse semantic, lexical and facet signals over a bounded vector candidate pool."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        place_repository: PlaceVectorRepository,
        weights: ContentRankingWeights | None = None,
        minimum_content_score: float = 0.20,
        k1: float = 1.5,
        b: float = 0.75,
        cache: SimpleTTLCache | None = None,
    ) -> None:
        if not 0 <= minimum_content_score <= 1:
            raise ValueError("minimum_content_score must be between zero and one")
        self._embedding_provider = embedding_provider
        self._place_repository = place_repository
        self._weights = weights or ContentRankingWeights()
        self._minimum_content_score = minimum_content_score
        self._k1 = k1
        self._b = b
        self._cache = cache

    async def retrieve(
        self,
        intent: ParsedPlaceChatIntent,
        limit: int,
    ) -> Sequence[PlaceChatCandidate]:
        if intent.action != "recommendations" or not intent.target_category:
            return []
        if limit < 1:
            return []

        semantic_query = prepare_for_embedding(intent.semantic_query)
        cache_key = self._cache_key(intent, semantic_query, limit)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        embedding = self._embedding_provider.embed_text(semantic_query)
        repository_categories = tuple(
            dict.fromkeys(
                (
                    *intent.category_values,
                    *intent.compatible_category_values,
                )
            )
        )
        filters = PlaceFilters(
            city=None,
            state=None,
            categories=repository_categories,
            price_range=_optional_string(intent.hard_filters.get("price_range")),
            occasion=_optional_string(intent.hard_filters.get("occasion")),
            is_active=True,
        )
        raw_candidates = list(
            await self._place_repository.search(
                embedding=embedding,
                filters=filters,
                limit=min(max(limit * 3, limit), 120),
            )
        )
        candidates = [
            candidate
            for candidate in raw_candidates
            if self._matches_hard_category(candidate, intent)
            and not self._matches_exclusions(candidate, intent.exclusions)
        ]
        if not candidates:
            return []

        corpus = [place_tokens(candidate) for candidate in candidates]
        query_tokens = tokenize(semantic_query)
        idf_index = idf_bm25(corpus)
        average_document_length = sum(map(len, corpus)) / len(corpus)
        ranked: list[PlaceChatCandidate] = []

        for candidate, document_tokens in zip(candidates, corpus):
            raw_lexical = bm25_score(
                document=document_tokens,
                query=query_tokens,
                idf_index=idf_index,
                average_document_length=average_document_length,
                k1=self._k1,
                b=self._b,
            )
            lexical = normalize_bm25(raw_lexical)
            semantic = _unit_score(candidate.score)
            theme_score, match_level, reasons = self._facet_match(
                candidate,
                intent,
                document_tokens,
            )
            content_score = self._weights.score(
                semantic_score=semantic,
                lexical_score=lexical,
                theme_or_reference_score=theme_score,
            )
            requires_content_evidence = bool(
                intent.soft_preferences
                or (intent.reference and intent.reference.entity)
            )
            if (
                requires_content_evidence
                and content_score < self._minimum_content_score
            ):
                continue
            ranked.append(
                PlaceChatCandidate(
                    place_id=candidate.id,
                    name=candidate.name,
                    category=candidate.category,
                    content_score=content_score,
                    semantic_score=semantic,
                    lexical_score=lexical,
                    match_level=match_level,
                    matched_reasons=reasons,
                    metadata=dict(candidate.metadata),
                )
            )

        ranked.sort(
            key=lambda item: (
                _match_level_priority(item.match_level),
                -item.content_score,
                -item.semantic_score,
                -item.lexical_score,
                item.place_id,
            )
        )
        unique_ranked: list[PlaceChatCandidate] = []
        seen_ids: set[str] = set()
        for candidate in ranked:
            if not candidate.place_id or candidate.place_id in seen_ids:
                continue
            seen_ids.add(candidate.place_id)
            unique_ranked.append(candidate)
            if len(unique_ranked) >= limit:
                break
        result = tuple(unique_ranked)
        if self._cache:
            self._cache.set(cache_key, result)
        return result

    @staticmethod
    def _cache_key(
        intent: ParsedPlaceChatIntent,
        semantic_query: str,
        limit: int,
    ) -> str:
        payload = {
            "query": semantic_query,
            "categories": intent.category_values,
            "compatible_categories": intent.compatible_category_values,
            "category_evidence_terms": intent.category_evidence_terms,
            "hard_filters": intent.hard_filters,
            "preferences": intent.soft_preferences,
            "exclusions": intent.exclusions,
            "limit": limit,
        }
        return "places-chat:" + json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
        )

    @staticmethod
    def _matches_hard_category(
        candidate: PlaceCandidate,
        intent: ParsedPlaceChatIntent,
    ) -> bool:
        if not candidate.category:
            return False
        actual = prepare_for_embedding(candidate.category).replace(" ", "_")
        exact = {
            prepare_for_embedding(value).replace(" ", "_")
            for value in intent.category_values
        }
        if actual in exact:
            return True
        compatible = {
            prepare_for_embedding(value).replace(" ", "_")
            for value in intent.compatible_category_values
        }
        if actual not in compatible:
            return False
        document_tokens = _category_evidence_tokens(candidate)
        return any(
            evidence_tokens and evidence_tokens <= document_tokens
            for term in intent.category_evidence_terms
            if (evidence_tokens := set(tokenize(term)))
        )

    @staticmethod
    def _matches_exclusions(
        candidate: PlaceCandidate,
        exclusions: tuple[str, ...],
    ) -> bool:
        if not exclusions:
            return False
        document_tokens = set(place_tokens(candidate))
        return any(
            exclusion_tokens
            and exclusion_tokens <= document_tokens
            for exclusion in exclusions
            if (
                exclusion_tokens := set(
                    tokenize(exclusion.replace("_", " "))
                )
            )
        )

    @staticmethod
    def _facet_match(
        candidate: PlaceCandidate,
        intent: ParsedPlaceChatIntent,
        document_tokens: list[str],
    ) -> tuple[float, str, tuple[str, ...]]:
        token_set = set(document_tokens)
        reasons: list[str] = [intent.target_category or "place"]
        matched_preferences = 0
        exact_preference = False
        for preference in intent.soft_preferences:
            preference_tokens = set(tokenize(preference.replace("_", " ")))
            if preference_tokens and preference_tokens <= token_set:
                matched_preferences += 1
                reasons.append(preference)
                exact_preference = (
                    exact_preference
                    or preference in _EXACT_ENTITY_PREFERENCES
                )

        preference_score = (
            matched_preferences / len(intent.soft_preferences)
            if intent.soft_preferences
            else 0.0
        )
        reference_exact = False
        if intent.reference and intent.reference.entity:
            reference_tokens = set(tokenize(intent.reference.entity))
            reference_exact = bool(reference_tokens and reference_tokens <= token_set)
            if reference_exact:
                reasons.append(intent.reference.entity)

        if reference_exact or exact_preference:
            return 1.0, "exact", tuple(dict.fromkeys(reasons))
        if matched_preferences:
            return preference_score, "family", tuple(dict.fromkeys(reasons))
        return 0.0, "broad", tuple(dict.fromkeys(reasons))


def _match_level_priority(level: str) -> int:
    return {"exact": 0, "family": 1, "broad": 2}.get(level, 3)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _category_evidence_tokens(candidate: PlaceCandidate) -> set[str]:
    # Do not inspect candidate.document here: indexed documents intentionally
    # contain broad category profiles (for example, all `entertainment` records
    # mention cinema). Compatibility must be proven by place-specific fields.
    evidence = " ".join(
        value
        for value in (
            candidate.name,
            _as_evidence_text(candidate.metadata.get("tags")),
            _as_evidence_text(candidate.metadata.get("short_description")),
        )
        if value
    )
    return set(tokenize(evidence))


def _as_evidence_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value)
    return str(value)


def _unit_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, score))
