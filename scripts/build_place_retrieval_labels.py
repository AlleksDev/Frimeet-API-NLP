from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import unicodedata


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.shared.nlp.embeddings.training_dataset import (
    dataset_validation_report,
    load_retrieval_training_dataset,
)


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "al", "algo", "con", "de", "del", "donde", "el", "en", "es",
    "la", "las", "lo", "los", "me", "mi", "para", "por", "que", "quiero",
    "un", "una", "y", "busco", "lugar", "lugares", "opcion", "opciones",
}
NAME_STOPWORDS = STOPWORDS | {
    "bar", "cafe", "cafeteria", "centro", "club", "comercial", "deportivo",
    "escuela", "farmacia", "hotel", "iglesia", "jardin", "museo", "parque",
    "restaurant", "restaurante", "tienda", "turistico", "turistica",
}


GENERIC_TEMPLATES = (
    "recomiéndame una opción de {label} {constraint}",
    "¿dónde puedo encontrar algo relacionado con {label} {constraint}?",
    "me interesa conocer opciones de {label} {constraint}",
    "necesito una recomendación de {label} {constraint}",
    "quiero descubrir un sitio de tipo {label} {constraint}",
    "muéstrame alternativas de {label} {constraint}",
    "ayúdame a encontrar una buena opción de {label} {constraint}",
    "estoy buscando específicamente {label} {constraint}",
    "quisiera una sugerencia relacionada con {label} {constraint}",
    "quiero comparar lugares de tipo {label} {constraint}",
    "necesito ubicar una opción de {label} {constraint}",
    "me gustaría visitar algo relacionado con {label} {constraint}",
    "quiero saber qué opciones de {label} hay disponibles {constraint}",
    "ayúdame a elegir un lugar relacionado con {label} {constraint}",
    "busco una recomendación local de {label} {constraint}",
    "quiero conocer una alternativa de {label} {constraint}",
    "qué lugar de tipo {label} podría visitar {constraint}",
    "dame una sugerencia concreta de {label} {constraint}",
    "estoy planeando una salida y necesito {label} {constraint}",
    "quiero encontrar en Chiapas una opción de {label} {constraint}",
    "qué alternativa de {label} existe en Chiapas {constraint}",
    "necesito información sobre lugares de {label} {constraint}",
    "me gustaría explorar una opción de {label} {constraint}",
    "busco algo que realmente corresponda a {label} {constraint}",
    "quiero una sugerencia para conocer {label} {constraint}",
    "qué sitio relacionado con {label} me recomiendas {constraint}",
    "quiero localizar una alternativa de {label} {constraint}",
    "dónde hay una opción de {label} que pueda conocer {constraint}",
    "necesito elegir entre lugares de {label} {constraint}",
    "me interesa encontrar una recomendación de {label} {constraint}",
    "quiero explorar lugares clasificados como {label} {constraint}",
    "qué recomendación tienes para una experiencia de {label} {constraint}",
    "busco una opción disponible de {label} {constraint}",
    "quiero conocer un sitio nuevo relacionado con {label} {constraint}",
    "ayúdame a ubicar algo de tipo {label} {constraint}",
    "necesito una alternativa local relacionada con {label} {constraint}",
)


TAG_PHRASES = {
    "accesibilidad universal": "con accesibilidad universal",
    "al aire libre": "al aire libre",
    "arte": "relacionado con arte",
    "bebidas": "con bebidas",
    "buffet": "con servicio de buffet",
    "cafe": "donde pueda tomar café",
    "cine": "con opciones de cine",
    "comida mexicana": "con comida mexicana",
    "comida rapida": "con opciones de comida rápida",
    "compras": "para hacer algunas compras",
    "cultural": "con una experiencia cultural",
    "deportes": "para realizar actividades deportivas",
    "divertido": "para pasar un rato divertido",
    "familiar": "para ir con mi familia",
    "lectura": "donde pueda leer",
    "mariscos": "con opciones de mariscos",
    "ninos": "para ir con niños",
    "nocturno": "para salir de noche",
    "para trabajar": "donde pueda trabajar un rato",
    "parrilla": "con comida a la parrilla",
    "pet friendly": "donde acepten mascotas",
    "postres": "con postres",
    "relajado": "en un ambiente relajado",
    "reuniones": "apropiado para reuniones",
    "tranquilo": "en un ambiente tranquilo",
    "turismo": "para conocer algo turístico",
    "vegano": "con opciones veganas",
    "vegetariano": "con opciones vegetarianas",
}


@dataclass(frozen=True)
class IntentProfile:
    key: str
    label: str
    category_terms: tuple[str, ...]
    templates: tuple[str, ...]
    allowed_tags: frozenset[str]
    confusion_groups: tuple[str, ...]


@dataclass(frozen=True)
class Place:
    place_id: str
    document: str
    name: str
    category: str
    tags: tuple[str, ...]
    source: str
    intent: str
    search_tokens: frozenset[str]
    normalized_tags: frozenset[str]
    name_tokens: frozenset[str]


@dataclass(frozen=True)
class LabeledQuery:
    query_id: str
    query: str
    positive: Place
    split: str
    used_name: bool


PROFILES = (
    IntentProfile(
        "health", "servicio de salud",
        ("farmacia", "fisioterapeuta", "dentista", "hospital", "salud", "medico", "pediatra", "laboratorio", "periodoncista", "veterinaria", "asesor medico", "vitaminas", "material sanitario", "medicamentos", "bienestar"),
        ("necesito encontrar un servicio de salud {constraint}", "busco atención o productos de salud {constraint}", "quiero ubicar un lugar relacionado con salud {constraint}", "necesito una opción de cuidado personal o médico {constraint}"),
        frozenset({"accesibilidad universal", "familiar"}),
        ("nightlife", "lodging", "religion"),
    ),
    IntentProfile(
        "education", "espacio educativo",
        ("escuela", "universidad", "instituto", "academia", "colegio", "centro educativo", "institucion educativa", "centro escolar", "centro de aprendizaje", "educacion", "jardin de infancia", "preescolar", "hauptschule", "formacion para obtener el carne"),
        ("busco un espacio educativo {constraint}", "necesito encontrar una escuela o centro de aprendizaje {constraint}", "quiero una opción para estudiar o aprender {constraint}", "busco una institución de formación {constraint}"),
        frozenset({"para trabajar", "lectura", "cultural", "arte", "ninos", "deportes"}),
        ("nightlife", "lodging", "health"),
    ),
    IntentProfile(
        "religion", "lugar religioso",
        ("iglesia", "capilla", "catedral", "parroquia", "congregacion", "lugar de culto", "institucion religiosa", "destino religioso"),
        ("busco un lugar religioso {constraint}", "quiero visitar una iglesia o templo {constraint}", "necesito encontrar un espacio de culto {constraint}", "busco un sitio para una actividad religiosa {constraint}"),
        frozenset({"familiar", "cultural", "tranquilo", "turismo"}),
        ("sports", "health", "shopping"),
    ),
    IntentProfile(
        "cafe", "cafetería",
        ("cafe", "cafeteria", "puesto de cafe", "tienda de cafe", "tostaderos de cafe", "mayorista de cafe"),
        ("quiero una cafetería {constraint}", "busco dónde tomar café {constraint}", "necesito un café para pasar un rato {constraint}", "quiero encontrar bebidas y algo ligero {constraint}", "busco una opción de café {constraint}"),
        frozenset({"tranquilo", "relajado", "para trabajar", "bebidas", "postres", "vegetariano", "vegano", "familiar", "accesibilidad universal", "pet friendly", "buffet"}),
        ("religion", "health", "education"),
    ),
    IntentProfile(
        "nightlife", "lugar de vida nocturna",
        ("bar", "cocteleria", "club nocturno", "cerveceria artesanal", "nightlife", "night club"),
        ("quiero salir por la noche {constraint}", "busco un bar o lugar para convivir {constraint}", "quiero tomar algo y pasarla bien {constraint}", "busco una opción de vida nocturna {constraint}", "quiero un lugar para salir con amigos {constraint}"),
        frozenset({"bebidas", "nocturno", "divertido", "nightlife", "night club", "parrilla", "comida rapida"}),
        ("religion", "education", "health"),
    ),
    IntentProfile(
        "restaurant", "restaurante",
        ("restaurant", "restaurante", "taqueria", "hamburgueseria", "marisqueria", "pizzeria", "parrilla", "buffet", "snack bar"),
        ("quiero un restaurante {constraint}", "busco un lugar para comer {constraint}", "necesito una opción para desayunar, almorzar o cenar {constraint}", "quiero probar comida {constraint}", "busco dónde comer con otras personas {constraint}"),
        frozenset({"familiar", "comida mexicana", "comida rapida", "vegetariano", "vegano", "mariscos", "parrilla", "buffet", "tranquilo", "bebidas", "postres"}),
        ("religion", "health", "education"),
    ),
    IntentProfile(
        "food_market", "tienda de alimentos",
        ("tienda de alimentacion", "carniceria", "supermercado", "panaderia", "pasteleria", "heladeria", "fruteria", "queseria", "abarrotes", "mercado", "chocolateria", "tienda de galletas", "tienda de postres", "tienda de tartas", "bufe de dulces", "bodega"),
        ("quiero comprar alimentos {constraint}", "busco una tienda de comida o productos básicos {constraint}", "necesito encontrar comestibles {constraint}", "quiero una opción para comprar ingredientes o alimentos {constraint}"),
        frozenset({"compras", "cafe", "postres", "vegetariano", "vegano", "mariscos", "parrilla"}),
        ("religion", "culture", "sports"),
    ),
    IntentProfile(
        "nature", "parque o espacio natural",
        ("park", "parque", "jardin", "senderismo", "fauna salvaje", "camping", "reserva", "zona de senderismo"),
        ("quiero un parque o espacio natural {constraint}", "busco dónde caminar y estar en contacto con la naturaleza {constraint}", "quiero pasar tiempo fuera de edificios {constraint}", "necesito un lugar con áreas verdes {constraint}", "busco una actividad en la naturaleza {constraint}"),
        frozenset({"al aire libre", "familiar", "tranquilo", "ninos", "deportes", "pet friendly", "turismo", "relajado"}),
        ("shopping", "health", "education"),
    ),
    IntentProfile(
        "sports", "lugar deportivo",
        ("sports", "gimnasio", "centro deportivo", "yoga", "taekwondo", "karate", "ciclismo", "piscina", "escuela deportiva", "deporte"),
        ("busco un lugar para hacer ejercicio {constraint}", "quiero practicar algún deporte {constraint}", "necesito un espacio de entrenamiento {constraint}", "busco una actividad física {constraint}"),
        frozenset({"al aire libre", "familiar", "ninos", "pet friendly", "accesibilidad universal"}),
        ("culture", "lodging", "religion"),
    ),
    IntentProfile(
        "culture", "lugar cultural",
        ("culture", "museo", "biblioteca", "planetario"),
        ("quiero conocer un lugar cultural {constraint}", "busco arte, historia o aprendizaje {constraint}", "quiero visitar un museo, biblioteca o centro cultural {constraint}", "necesito una actividad cultural {constraint}", "busco un espacio para aprender y explorar {constraint}"),
        frozenset({"arte", "lectura", "turismo", "tranquilo", "familiar", "ninos", "cine"}),
        ("sports", "health", "shopping"),
    ),
    IntentProfile(
        "lodging", "alojamiento",
        ("lodging", "hotel", "alojamiento", "albergue", "estancia en granjas", "condominio"),
        ("busco alojamiento {constraint}", "necesito un lugar donde hospedarme {constraint}", "quiero encontrar un hotel o estancia {constraint}", "busco dónde quedarme durante un viaje {constraint}"),
        frozenset({"turismo", "tranquilo", "familiar", "bebidas", "para trabajar", "accesibilidad universal", "pet friendly"}),
        ("health", "education", "shopping"),
    ),
    IntentProfile(
        "tourism", "atracción turística",
        ("tourism", "atraccion turistica", "hacienda turistica"),
        ("quiero conocer una atracción turística {constraint}", "busco algo interesante para visitar {constraint}", "quiero explorar un sitio turístico {constraint}", "necesito una actividad para conocer Chiapas {constraint}"),
        frozenset({"cultural", "arte", "al aire libre", "familiar", "tranquilo"}),
        ("health", "education", "shopping"),
    ),
    IntentProfile(
        "community", "espacio comunitario o de entretenimiento",
        ("community", "entertainment", "family", "parque infantil", "asociacion", "salon para eventos", "club de ajedrez", "edificio multiusos", "cibercafe", "ciber"),
        ("busco un espacio para convivir {constraint}", "quiero una actividad comunitaria o de entretenimiento {constraint}", "necesito un lugar para reunirme con otras personas {constraint}", "busco un plan para compartir tiempo {constraint}"),
        frozenset({"familiar", "ninos", "divertido", "cine", "reuniones", "cultural", "arte", "tranquilo"}),
        ("health", "lodging", "religion"),
    ),
    IntentProfile(
        "shopping", "lugar de compras",
        ("shopping", "tienda", "comercio", "papeleria", "bazar", "casa de empenos", "centro comercial", "zapateria", "perfumeria", "colchoneria"),
        ("quiero ir de compras {constraint}", "busco una tienda o comercio {constraint}", "necesito encontrar productos para comprar {constraint}", "quiero visitar una zona comercial {constraint}"),
        frozenset({"familiar", "ninos", "accesibilidad universal", "cafe", "cine"}),
        ("religion", "sports", "lodging"),
    ),
)


PROFILE_BY_KEY = {profile.key: profile for profile in PROFILES}
FOOD_GROUPS = {"cafe", "restaurant", "food_market", "nightlife"}


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)
    places, unmapped_categories = _load_places(Path(args.input))
    places_by_intent: dict[str, list[Place]] = defaultdict(list)
    for place in places:
        places_by_intent[place.intent].append(place)

    labeled_queries: list[LabeledQuery] = []
    used_queries: set[str] = set()
    query_serial = 1
    for profile in PROFILES:
        candidates = places_by_intent.get(profile.key, [])
        selected = _select_positives(candidates, args.queries_per_intent, rng)
        group_queries: list[LabeledQuery] = []
        for index, positive in enumerate(selected):
            query, used_name = _build_unique_query(
                profile=profile,
                positive=positive,
                index=index,
                used_queries=used_queries,
            )
            if query is None:
                continue
            normalized_query = _normalize(query)
            used_queries.add(normalized_query)
            group_queries.append(
                LabeledQuery(
                    query_id=f"q-{profile.key}-{query_serial:04d}",
                    query=query,
                    positive=positive,
                    split="",
                    used_name=used_name,
                )
            )
            query_serial += 1

        rng.shuffle(group_queries)
        split_counts = _split_counts(len(group_queries))
        for index, item in enumerate(group_queries):
            split = (
                "train" if index < split_counts[0]
                else "validation" if index < split_counts[0] + split_counts[1]
                else "test"
            )
            labeled_queries.append(
                LabeledQuery(
                    query_id=item.query_id,
                    query=item.query,
                    positive=item.positive,
                    split=split,
                    used_name=item.used_name,
                )
            )

    rows: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []
    negative_type_counts: Counter[str] = Counter()
    for item in labeled_queries:
        negatives = _select_negatives(
            item=item,
            places=places,
            places_by_intent=places_by_intent,
            count=args.negatives_per_query,
        )
        for negative, negative_type in negatives:
            row = {
                "query_id": item.query_id,
                "query": item.query,
                "positive_id": item.positive.place_id,
                "positive": item.positive.document,
                "negative_id": negative.place_id,
                "negative": negative.document,
                "negative_type": negative_type,
                "split": item.split,
            }
            rows.append(row)
            negative_type_counts[negative_type] += 1
            review_rows.append(
                {
                    "query_id": item.query_id,
                    "split": item.split,
                    "query": item.query,
                    "positive_id": item.positive.place_id,
                    "positive_name": item.positive.name,
                    "positive_category": item.positive.category,
                    "positive_tags": ",".join(item.positive.tags),
                    "negative_id": negative.place_id,
                    "negative_name": negative.name,
                    "negative_category": negative.category,
                    "negative_tags": ",".join(negative.tags),
                    "negative_type": negative_type,
                    "review_status": "pending",
                    "reviewer_notes": "",
                }
            )

    output_path = Path(args.output).resolve()
    review_path = Path(args.review_csv).resolve()
    report_path = Path(args.report).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path, rows)
    _write_review_csv(review_path, review_rows)

    validated = load_retrieval_training_dataset(output_path)
    validation = dataset_validation_report(validated)
    report = {
        "input": str(Path(args.input).resolve()),
        "output": str(output_path),
        "review_csv": str(review_path),
        "corpus_rows": len(places) + sum(unmapped_categories.values()),
        "mapped_places": len(places),
        "unmapped_places": sum(unmapped_categories.values()),
        "unmapped_categories": dict(unmapped_categories.most_common()),
        "mapped_places_by_intent": {
            key: len(value) for key, value in sorted(places_by_intent.items())
        },
        "selected_queries_by_intent": dict(
            sorted(Counter(item.positive.intent for item in labeled_queries).items())
        ),
        "name_based_queries": sum(item.used_name for item in labeled_queries),
        "negative_types": dict(sorted(negative_type_counts.items())),
        "validation": {
            "total_rows": validation.total_rows,
            "rows_by_split": validation.rows_by_split,
            "unique_queries_by_split": validation.unique_queries_by_split,
        },
        "warnings": [
            "These are weak labels generated from category and tags; review every row marked pending before training.",
            "No spatial-reference examples were generated because the corpus has no verified coordinates or proximity relationships.",
            "Missing tags mean that absence of a constraint is not treated automatically as a negative label.",
        ],
        "seed": args.seed,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _load_places(path: Path) -> tuple[list[Place], Counter[str]]:
    raw_rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            raw_rows.append(payload)

    places: list[Place] = []
    unmapped: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for payload in raw_rows:
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict) or metadata.get("is_active") is False:
            continue
        place_id = str(payload.get("place_id") or "").strip()
        document = str(payload.get("document") or "").strip()
        name = str(metadata.get("name") or "").strip()
        category = str(metadata.get("category") or "").strip()
        if not place_id or not document or not name or place_id in seen_ids:
            continue
        seen_ids.add(place_id)
        intent = _classify_category(category)
        if intent is None:
            unmapped[category or "<empty>"] += 1
            continue
        tags = tuple(
            tag.strip()
            for tag in str(metadata.get("tags") or "").split(",")
            if tag.strip()
        )
        normalized_tags = frozenset(_normalize(tag) for tag in tags)
        places.append(
            Place(
                place_id=place_id,
                document=document,
                name=name,
                category=category,
                tags=tags,
                source=str(metadata.get("source") or ""),
                intent=intent,
                search_tokens=frozenset(
                    _tokens(f"{name} {category} {' '.join(tags)}")
                ),
                normalized_tags=normalized_tags,
                name_tokens=frozenset(_name_tokens(name)),
            )
        )
    return places, unmapped


def _classify_category(category: str) -> str | None:
    normalized = _normalize(category)
    for profile in PROFILES:
        if any(_term_matches(normalized, term) for term in profile.category_terms):
            return profile.key
    return None


def _term_matches(category: str, term: str) -> bool:
    normalized_term = _normalize(term)
    if normalized_term in {"bar", "cafe", "park", "hotel", "restaurant", "sports", "culture", "shopping", "tourism", "lodging", "community", "family", "entertainment"}:
        return category == normalized_term or re.search(
            rf"\b{re.escape(normalized_term)}\b", category
        ) is not None
    return normalized_term in category


def _select_positives(
    places: list[Place],
    limit: int,
    rng: random.Random,
) -> list[Place]:
    tagged = [place for place in places if place.tags]
    untagged = [place for place in places if not place.tags]
    rng.shuffle(tagged)
    rng.shuffle(untagged)
    tagged_target = min(len(tagged), max(1, round(limit * 0.8)))
    selected = tagged[:tagged_target]
    selected.extend(untagged[: max(0, limit - len(selected))])
    if len(selected) < limit:
        selected.extend(tagged[tagged_target : tagged_target + limit - len(selected)])
    return selected[:limit]


def _build_unique_query(
    profile: IntentProfile,
    positive: Place,
    index: int,
    used_queries: set[str],
) -> tuple[str | None, bool]:
    constraints = _constraint_phrases(profile, positive)
    constraint_variants = [""]
    constraint_variants.extend(constraints)
    if len(constraints) >= 2:
        constraint_variants.extend(
            f"{constraints[left]} y {constraints[right]}"
            for left in range(len(constraints))
            for right in range(left + 1, len(constraints))
        )

    if index % 10 == 9:
        name_candidate = _clean_query(
            f"quiero visitar {positive.name} o un {profile.label} de ese estilo"
        )
        if _normalize(name_candidate) not in used_queries:
            return name_candidate, True

    templates = profile.templates + GENERIC_TEMPLATES
    for offset in range(len(templates) * max(1, len(constraint_variants))):
        template = templates[(index + offset) % len(templates)]
        constraint = constraint_variants[
            (index + offset // len(templates)) % len(constraint_variants)
        ]
        candidate = _clean_query(
            template.format(label=profile.label, constraint=constraint)
        )
        if _normalize(candidate) not in used_queries:
            return candidate, False

    name_queries = (
        f"quiero visitar {positive.name} o un {profile.label} de ese estilo",
        f"busco un lugar parecido a {positive.name} que funcione como {profile.label}",
        f"me interesa {positive.name} y quiero una experiencia de {profile.label}",
    )
    for candidate in name_queries:
        candidate = _clean_query(candidate)
        if _normalize(candidate) not in used_queries:
            return candidate, True
    return None, False


def _constraint_phrases(profile: IntentProfile, place: Place) -> list[str]:
    phrases: list[str] = []
    for tag in place.tags:
        normalized_tag = _normalize(tag)
        if normalized_tag not in profile.allowed_tags:
            continue
        phrase = TAG_PHRASES.get(normalized_tag)
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return phrases[:3]


def _split_counts(total: int) -> tuple[int, int, int]:
    if total < 3:
        return total, 0, 0
    validation = max(1, round(total * 0.1))
    test = max(1, round(total * 0.1))
    train = total - validation - test
    return train, validation, test


def _select_negatives(
    item: LabeledQuery,
    places: list[Place],
    places_by_intent: dict[str, list[Place]],
    count: int,
) -> list[tuple[Place, str]]:
    profile = PROFILE_BY_KEY[item.positive.intent]
    selected: list[tuple[Place, str]] = []
    selected_ids: set[str] = {item.positive.place_id}

    if item.positive.intent not in FOOD_GROUPS:
        bias_candidates = [
            place
            for group in ("restaurant", "cafe")
            for place in places_by_intent.get(group, [])
        ]
        candidate = _best_negative(
            query=item.query,
            positive=item.positive,
            candidates=bias_candidates,
            excluded_ids=selected_ids,
        )
        if candidate is not None:
            selected.append((candidate, "popular_category_bias"))
            selected_ids.add(candidate.place_id)
            if len(selected) >= count:
                return selected

    for confusion_group in profile.confusion_groups:
        candidate = _best_negative(
            query=item.query,
            positive=item.positive,
            candidates=places_by_intent.get(confusion_group, []),
            excluded_ids=selected_ids,
        )
        if candidate is not None:
            selected.append((candidate, _negative_type(item, candidate)))
            selected_ids.add(candidate.place_id)
            if len(selected) >= count:
                return selected

    if len(selected) < count:
        query_tokens = _tokens(item.query)
        fallback = sorted(
            (
                place for place in places
                if place.intent != item.positive.intent and place.place_id not in selected_ids
            ),
            key=lambda place: (
                -_negative_score(query_tokens, item.positive, place),
                _stable_tie_break(item.query, place.place_id),
            ),
        )
        for candidate in fallback:
            selected.append((candidate, _negative_type(item, candidate)))
            selected_ids.add(candidate.place_id)
            if len(selected) >= count:
                break
    return selected


def _best_negative(
    query: str,
    positive: Place,
    candidates: list[Place],
    excluded_ids: set[str],
) -> Place | None:
    available = [place for place in candidates if place.place_id not in excluded_ids]
    if not available:
        return None
    query_tokens = _tokens(query)
    return min(
        available,
        key=lambda place: (
            -_negative_score(query_tokens, positive, place),
            _stable_tie_break(query, place.place_id),
        ),
    )


def _negative_score(
    query_tokens: set[str],
    positive: Place,
    candidate: Place,
) -> float:
    lexical_overlap = len(query_tokens & candidate.search_tokens)
    tag_overlap = len(positive.normalized_tags & candidate.normalized_tags)
    name_overlap = len(positive.name_tokens & candidate.name_tokens)
    return lexical_overlap * 4.0 + tag_overlap * 2.5 + name_overlap * 3.0


def _negative_type(item: LabeledQuery, candidate: Place) -> str:
    query_overlap = _tokens(item.query) & candidate.name_tokens
    positive_name_overlap = item.positive.name_tokens & candidate.name_tokens
    if item.used_name and (query_overlap or positive_name_overlap):
        return "ambiguous_name"
    if query_overlap:
        return "lexical_overlap_wrong_intent"
    return "category_confusion"


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No labels were generated")
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _tokens(value: str) -> set[str]:
    return {
        token for token in TOKEN_PATTERN.findall(_normalize(value))
        if token not in STOPWORDS and len(token) > 1
    }


def _name_tokens(value: str) -> set[str]:
    return {
        token for token in TOKEN_PATTERN.findall(_normalize(value))
        if token not in NAME_STOPWORDS and len(token) > 2
    }


def _stable_tie_break(query: str, place_id: str) -> str:
    return hashlib.sha256(f"{query}|{place_id}".encode("utf-8")).hexdigest()


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value).casefold())
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_accents).strip()


def _clean_query(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" ,.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build human-reviewable weak labels from a real place corpus."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--queries-per-intent", type=int, default=40)
    parser.add_argument("--negatives-per-query", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
