from app.modules.places.application.use_cases.evaluate_place_search import (
    PlaceSearchEvaluationCase,
)
from app.modules.places.domain.models import PlaceFilters


BENCHMARK_NAME = "built_in_places_v1"
QRELS_SOURCE = "predefined_graded_qrels"


def get_default_place_search_benchmark() -> list[PlaceSearchEvaluationCase]:
    active_places = PlaceFilters(is_active=True)
    return [
        PlaceSearchEvaluationCase(
            query="cafe tranquilo con postres",
            relevance={"place_1": 3},
            filters=active_places,
        ),
        PlaceSearchEvaluationCase(
            query="mirador para fotos al atardecer",
            relevance={"place_2": 3},
            filters=active_places,
        ),
        PlaceSearchEvaluationCase(
            query="comida regional en restaurante",
            relevance={"place_3": 3, "place_4": 1},
            filters=active_places,
        ),
        PlaceSearchEvaluationCase(
            query="cena tranquila en pareja",
            relevance={"place_4": 3, "place_1": 1},
            filters=active_places,
        ),
        PlaceSearchEvaluationCase(
            query="plan con amigos para comer o tomar cafe",
            relevance={"place_1": 3, "place_3": 2},
            filters=active_places,
        ),
    ]
