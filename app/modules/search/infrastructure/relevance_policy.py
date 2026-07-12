from app.modules.search.domain.models import SearchResourceType
from app.modules.search.domain.relevance import (
    SearchRelevancePolicy,
    SearchRelevanceThreshold,
)
from app.shared.config.settings import Settings


def build_relevance_policy(settings: Settings) -> SearchRelevancePolicy:
    thresholds: dict[SearchResourceType, SearchRelevanceThreshold] = {}
    for resource_type in SearchResourceType:
        override = settings.global_search_resource_thresholds.get(
            resource_type.value,
            {},
        )
        thresholds[resource_type] = SearchRelevanceThreshold(
            semantic_min=float(
                override.get(
                    "semantic_min",
                    settings.global_search_min_semantic_score,
                )
            ),
            lexical_min=float(
                override.get(
                    "lexical_min",
                    settings.global_search_min_lexical_score,
                )
            ),
        )
    return SearchRelevancePolicy(
        thresholds=thresholds,
        version=settings.global_search_threshold_policy_version,
    )
