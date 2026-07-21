import asyncio
from collections.abc import Mapping
import json
import math
import re
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
_ATTRIBUTE_PREFERENCE_KEYS: dict[str, tuple[str, ...]] = {
    "estacionamiento": ("has_parking",),
    "banos": ("has_restrooms",),
    "para_refrescarse": ("good_for_cooling_off",),
    "entretenimiento": ("has_entertainment",),
    "musica": ("has_entertainment",),
    "pantallas": ("has_screens",),
    "asientos": ("has_seating",),
    "area_infantil": ("has_kids_playroom",),
    "para_amigos": ("has_friends_space",),
    "comida_exterior": ("allows_outside_food",),
    "mochila": ("allows_backpack",),
    "romantico": ("has_romantic_space",),
    "familiar": ("has_family_space", "kid_friendly"),
}
_POSITIVE_ATTRIBUTE_STATES = {True, "true", "yes", "normal"}
_NEGATIVE_ATTRIBUTE_STATES = {False, "false", "no"}


class HybridContentPlaceChatRetriever:
    """Fuse semantic, lexical and facet signals over a bounded vector candidate pool.

    Categories are deliberately treated as ranking evidence, not repository
    filters.  This keeps open-vocabulary queries useful even when the parser
    cannot map the user's wording to a canonical category.
    """

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
        if intent.action != "recommendations":
            return []
        if limit < 1:
            return []

        semantic_query = prepare_for_embedding(intent.semantic_query)
        cache_key = self._cache_key(intent, semantic_query, limit)
        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        embedding = await asyncio.to_thread(
            self._embedding_provider.embed_text,
            semantic_query,
        )
        filters = PlaceFilters(
            city=_optional_string(intent.hard_filters.get("city")),
            state=_optional_string(intent.hard_filters.get("state")),
            # A canonical category is a hypothesis, not a hard constraint.  A
            # category filter here would prevent semantic retrieval from ever
            # seeing useful places whose source taxonomy differs from ours.
            categories=None,
            price_range=_optional_string(intent.hard_filters.get("price_range")),
            occasion=_optional_string(intent.hard_filters.get("occasion")),
            place_ids=(
                tuple(
                    str(place_id)
                    for place_id in intent.hard_filters.get("place_ids", ())
                    if str(place_id).strip()
                )
                if intent.hard_filters.get("place_ids") is not None
                else None
            ),
            is_active=True,
        )
        candidate_pool_limit = min(max(limit * 8, 40), 120)
        raw_candidates = list(
            await self._search_repository(
                query_text=intent.semantic_query.strip(),
                embedding=embedding,
                filters=filters,
                limit=candidate_pool_limit,
            )
        )
        candidates = raw_candidates
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
            repository_semantic = _optional_finite_score(
                candidate.metadata.get("semantic_score")
            )
            repository_lexical = _optional_finite_score(
                candidate.metadata.get("lexical_score")
            )
            has_repository_channel_scores = (
                repository_semantic is not None or repository_lexical is not None
            )
            semantic = _unit_score(
                repository_semantic
                if has_repository_channel_scores
                else candidate.score
            )
            if repository_lexical is not None:
                lexical = max(lexical, normalize_bm25(repository_lexical))
            category_score, category_match, category_reason = (
                self._category_affinity(candidate, intent)
            )
            theme_score, match_level, reasons = self._facet_match(
                candidate,
                intent,
                document_tokens,
                category_score=category_score,
                category_reason=category_reason,
            )
            base_content_score = self._weights.score(
                semantic_score=semantic,
                lexical_score=lexical,
                theme_or_reference_score=theme_score,
            )
            exclusion_affinity, exclusion_matches = self._exclusion_affinity(
                candidate,
                intent.exclusions,
            )
            attribute_conflict_affinity, attribute_conflicts = (
                self._attribute_conflict_affinity(candidate, intent)
            )
            # Exclusions are negative ranking evidence, not a boolean gate. A
            # source description may mention an excluded concept in a negated
            # form ("sin ruido"), and removing that row would invert intent.
            # Explicit false/no attribute values are also a modest penalty.
            # Missing/NULL values never enter this calculation and stay neutral.
            content_score = max(
                0.0,
                base_content_score
                - 0.35 * exclusion_affinity
                - 0.18 * attribute_conflict_affinity,
            )
            effective_minimum_content_score = (
                self._effective_minimum_content_score(candidate)
            )
            # Keep weak candidates so short or novel queries do not collapse to
            # zero results.  The old minimum remains a quality diagnostic for
            # downstream confidence/clarification policy, rather than a gate.
            metadata = dict(candidate.metadata)
            diagnostics = {
                "category_affinity": round(category_score, 6),
                "category_match": category_match,
                "exclusion_affinity": round(exclusion_affinity, 6),
                "exclusion_matches": list(exclusion_matches),
                "attribute_conflict_affinity": round(
                    attribute_conflict_affinity,
                    6,
                ),
                "attribute_conflicts": list(attribute_conflicts),
                "content_quality": (
                    "sufficient"
                    if content_score >= effective_minimum_content_score
                    else "weak"
                ),
                "meets_minimum_content_score": (
                    content_score >= effective_minimum_content_score
                ),
                "minimum_content_score": effective_minimum_content_score,
                "query_token_count": len(query_tokens),
            }
            if effective_minimum_content_score != self._minimum_content_score:
                diagnostics["configured_minimum_content_score"] = (
                    self._minimum_content_score
                )
            metadata["retrieval_diagnostics"] = diagnostics
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
                    metadata=metadata,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item.content_score,
                _attribute_conflict_sort_value(item),
                -item.semantic_score,
                -item.lexical_score,
                _match_level_priority(item.match_level),
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

    def _effective_minimum_content_score(
        self,
        candidate: PlaceCandidate,
    ) -> float:
        """Avoid treating unknown optional fields as negative relevance.

        New sync records explicitly list which source fields were present.  A
        sparse record still needs positive evidence, but it is evaluated
        against a slightly lower decision threshold instead of losing points
        merely because its description or tags are unknown.
        """

        raw_fields = candidate.metadata.get("source_fields_present")
        if not isinstance(raw_fields, (list, tuple, set)):
            return self._minimum_content_score
        fields = {str(field).strip().casefold() for field in raw_fields}
        missing_optional = sum(
            field not in fields for field in ("description", "tags")
        )
        if missing_optional == 0:
            return self._minimum_content_score
        return max(
            0.12,
            self._minimum_content_score - 0.035 * missing_optional,
        )

    async def _search_repository(
        self,
        query_text: str,
        embedding: list[float],
        filters: PlaceFilters,
        limit: int,
    ) -> Sequence[PlaceCandidate]:
        """Prefer an optional independent dense+lexical repository search.

        ``PlaceVectorRepository`` intentionally keeps its existing contract.
        Repositories can opt into hybrid candidate generation duck-typically,
        while mocks and current adapters continue through ``search``.
        """

        hybrid_search = getattr(self._place_repository, "search_hybrid", None)
        if callable(hybrid_search):
            return await hybrid_search(
                query_text=query_text,
                embedding=embedding,
                filters=filters,
                limit=limit,
            )
        return await self._place_repository.search(
            embedding=embedding,
            filters=filters,
            limit=limit,
        )

    @staticmethod
    def _cache_key(
        intent: ParsedPlaceChatIntent,
        semantic_query: str,
        limit: int,
    ) -> str:
        payload = {
            "query": semantic_query,
            "target_category": intent.target_category,
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
    def _category_affinity(
        candidate: PlaceCandidate,
        intent: ParsedPlaceChatIntent,
    ) -> tuple[float, str, str | None]:
        actual = _normalized_category(candidate.category)
        exact = {
            normalized
            for value in (intent.target_category, *intent.category_values)
            if (normalized := _normalized_category(value))
        }
        compatible = {
            normalized
            for value in intent.compatible_category_values
            if (normalized := _normalized_category(value))
        }
        has_category_hypothesis = bool(exact or compatible)
        if not has_category_hypothesis:
            return 0.0, "not_requested", None

        reason = intent.target_category or next(iter(intent.category_values), None)
        if actual and actual in exact:
            return 1.0, "exact", reason or candidate.category

        candidate_evidence_tokens = _category_evidence_tokens(candidate)
        has_specific_evidence = any(
            term_tokens and term_tokens <= candidate_evidence_tokens
            for term in intent.category_evidence_terms
            if (term_tokens := set(tokenize(term)))
        )
        if actual and actual in compatible:
            if has_specific_evidence:
                return 0.80, "compatible_with_evidence", reason
            return 0.35, "compatible", reason
        if has_specific_evidence:
            return 0.65, "textual_evidence", reason
        return 0.0, "none", None

    @staticmethod
    def _exclusion_affinity(
        candidate: PlaceCandidate,
        exclusions: tuple[str, ...],
    ) -> tuple[float, tuple[str, ...]]:
        if not exclusions:
            return 0.0, ()
        evidence = prepare_for_embedding(
            " ".join(
                value
                for value in (
                    candidate.name,
                    candidate.category or "",
                    candidate.document or "",
                    _as_evidence_text(candidate.metadata.get("tags")),
                    _as_evidence_text(candidate.metadata.get("short_description")),
                    _positive_facet_evidence(candidate),
                )
                if value
            )
        )
        document_tokens = set(tokenize(evidence))
        matches: list[str] = []
        for exclusion in exclusions:
            normalized = prepare_for_embedding(exclusion.replace("_", " "))
            exclusion_tokens = set(tokenize(normalized))
            if not exclusion_tokens or not exclusion_tokens <= document_tokens:
                continue
            if _is_negated_in_evidence(evidence, normalized):
                continue
            matches.append(exclusion)
        unique_matches = tuple(dict.fromkeys(matches))
        return len(unique_matches) / len(exclusions), unique_matches

    @staticmethod
    def _attribute_conflict_affinity(
        candidate: PlaceCandidate,
        intent: ParsedPlaceChatIntent,
    ) -> tuple[float, tuple[str, ...]]:
        raw_states = candidate.metadata.get("attribute_states")
        if not isinstance(raw_states, Mapping):
            return 0.0, ()

        known_preferences = 0
        conflicts: list[str] = []
        for preference in intent.soft_preferences:
            canonical = str(preference).strip().casefold()
            keys = _ATTRIBUTE_PREFERENCE_KEYS.get(canonical)
            if keys is None and canonical == "economico":
                price_tier = _normalized_attribute_state(
                    raw_states.get("price_tier")
                )
                if not price_tier:
                    continue
                known_preferences += 1
                if price_tier in {"expensive", "very_expensive", "deluxe"}:
                    conflicts.append(canonical)
                continue
            if not keys:
                continue

            values = tuple(
                _normalized_attribute_state(raw_states.get(key))
                for key in keys
                if key in raw_states
            )
            values = tuple(value for value in values if value != "")
            if not values:
                continue
            known_preferences += 1
            if any(value in _POSITIVE_ATTRIBUTE_STATES for value in values):
                continue
            if any(value in _NEGATIVE_ATTRIBUTE_STATES for value in values):
                conflicts.append(canonical)

        unique_conflicts = tuple(dict.fromkeys(conflicts))
        if known_preferences == 0:
            return 0.0, unique_conflicts
        return len(unique_conflicts) / known_preferences, unique_conflicts

    @staticmethod
    def _positive_attribute_preferences(
        candidate: PlaceCandidate,
        preferences: tuple[str, ...],
    ) -> set[str]:
        raw_states = candidate.metadata.get("attribute_states")
        if not isinstance(raw_states, Mapping):
            return set()

        matched: set[str] = set()
        for preference in preferences:
            canonical = str(preference).strip().casefold()
            if canonical == "economico":
                if _normalized_attribute_state(raw_states.get("price_tier")) in {
                    "cheap",
                    "accessible",
                }:
                    matched.add(canonical)
                continue
            # Generic entertainment does not prove that music is available;
            # that preference still needs a feature/note lexical match.
            if canonical == "musica":
                continue
            keys = _ATTRIBUTE_PREFERENCE_KEYS.get(canonical, ())
            values = (
                _normalized_attribute_state(raw_states.get(key))
                for key in keys
                if key in raw_states
            )
            if any(value in _POSITIVE_ATTRIBUTE_STATES for value in values):
                matched.add(canonical)
        return matched

    @staticmethod
    def _facet_match(
        candidate: PlaceCandidate,
        intent: ParsedPlaceChatIntent,
        document_tokens: list[str],
        category_score: float = 0.0,
        category_reason: str | None = None,
    ) -> tuple[float, str, tuple[str, ...]]:
        token_set = set(document_tokens)
        reasons: list[str] = []
        if category_score > 0 and category_reason:
            reasons.append(category_reason)
        matched_preferences = 0
        exact_preference = False
        structured_matches = (
            HybridContentPlaceChatRetriever._positive_attribute_preferences(
                candidate,
                intent.soft_preferences,
            )
        )
        for preference in intent.soft_preferences:
            preference_tokens = set(tokenize(preference.replace("_", " ")))
            if (
                preference in structured_matches
                or (preference_tokens and preference_tokens <= token_set)
            ):
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

        facet_score = max(
            category_score,
            preference_score,
            1.0 if reference_exact else 0.0,
        )
        if reference_exact or exact_preference or category_score >= 1.0:
            return facet_score, "exact", tuple(dict.fromkeys(reasons))
        if matched_preferences or category_score >= 0.50:
            return facet_score, "family", tuple(dict.fromkeys(reasons))
        return facet_score, "broad", tuple(dict.fromkeys(reasons))


def _match_level_priority(level: str) -> int:
    return {"exact": 0, "family": 1, "broad": 2}.get(level, 3)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalized_category(value: object) -> str:
    if value is None:
        return ""
    return prepare_for_embedding(str(value)).replace(" ", "_")


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
            _positive_facet_evidence(candidate),
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


def _normalized_attribute_state(value: object) -> bool | str:
    if isinstance(value, bool):
        return value
    if value is None:
        return ""
    return str(value).strip().casefold()


def _positive_facet_evidence(candidate: PlaceCandidate) -> str:
    return " ".join(
        _as_evidence_text(candidate.metadata.get(key))
        for key in (
            "attribute_terms",
            "entertainment_features",
            "contained_items",
            "menu_items",
        )
    )


def _attribute_conflict_sort_value(candidate: PlaceChatCandidate) -> float:
    diagnostics = candidate.metadata.get("retrieval_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return 0.0
    value = diagnostics.get("attribute_conflict_affinity")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return max(0.0, float(value))


def _is_negated_in_evidence(evidence: str, concept: str) -> bool:
    """Detect common source-side negation without business-category rules."""

    escaped = re.escape(concept).replace(r"\ ", r"\s+")
    negation = (
        r"(?:sin|libre\s+de|no\s+(?:hay|tiene|ofrece)|"
        r"evita(?:r)?|prohibid[oa]s?)"
    )
    return bool(
        re.search(
            rf"\b{negation}\s+(?:\w+\s+){{0,2}}{escaped}\b",
            evidence,
        )
    )


def _unit_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(1.0, score))


def _optional_finite_score(value: object) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None
