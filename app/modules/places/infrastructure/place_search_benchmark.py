from app.modules.places.application.use_cases.evaluate_place_search import (
    PlaceSearchEvaluationCase,
)
from app.modules.places.domain.models import PlaceFilters


BENCHMARK_NAME = "built_in_places_v2"
QRELS_SOURCE = "predefined_graded_qrels"


def get_default_place_search_benchmark() -> list[PlaceSearchEvaluationCase]:
    active_places = PlaceFilters(is_active=True)
    return [
        PlaceSearchEvaluationCase(
            query="quiero un lugar calmado donde pueda leer un rato sin mucho ruido",
            relevance={"place_5": 3, "place_1": 1},
            filters=active_places,
        ),
        PlaceSearchEvaluationCase(
            query="busco un plan barato con mi pareja para ver algo bonito y tomar fotos",
            relevance={"place_2": 3, "place_12": 2, "place_8": 1},
            filters=active_places,
        ),
        PlaceSearchEvaluationCase(
            query="mis papas vienen de visita y quiero llevarlos a probar algo muy chiapaneco",
            relevance={"place_3": 3, "place_11": 2},
            filters=active_places,
        ),
        PlaceSearchEvaluationCase(
            query="donde puedo hacer ejercicio temprano y llevar a mi perro",
            relevance={"place_6": 3, "place_8": 1},
            filters=active_places,
        ),
        PlaceSearchEvaluationCase(
            query="quiero conocer un poco de arte e historia sin estar todo el tiempo afuera",
            relevance={"place_10": 3, "place_7": 3},
            filters=active_places,
        ),
        PlaceSearchEvaluationCase(
            query="necesito una idea sencilla para una cita sin gastar demasiado",
            relevance={"place_2": 3, "place_1": 2, "place_8": 2, "place_12": 1},
            filters=active_places,
        ),
        PlaceSearchEvaluationCase(
            query="esta lloviendo y queremos salir varios amigos a algun lugar bajo techo",
            relevance={"place_9": 3, "place_1": 2, "place_7": 1, "place_10": 1},
            filters=active_places,
        ),
        PlaceSearchEvaluationCase(
            query="quiero despejarme caminando entre arboles y lejos del trafico",
            relevance={"place_12": 3, "place_8": 3, "place_6": 1},
            filters=active_places,
        ),
        PlaceSearchEvaluationCase(
            query="necesito sentarme con la laptop unas horas y tomar algo mientras trabajo",
            relevance={"place_1": 3, "place_5": 2},
            filters=active_places,
        ),
        PlaceSearchEvaluationCase(
            query="quiero una cena especial para aniversario en un ambiente relajado",
            relevance={"place_4": 3, "place_3": 2},
            filters=active_places,
        ),
    ]
