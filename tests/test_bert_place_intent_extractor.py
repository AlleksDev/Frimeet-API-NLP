import math

import pytest

from app.modules.places.infrastructure.bert_intent_extractor import (
    BertIntentInferenceError,
    BertIntentModelLoadError,
    BertIntentOutputError,
    BertPlaceIntentExtractor,
)


class FakeTokenClassifier:
    def __init__(self, predictions: list[dict[str, object]]) -> None:
        self.predictions = predictions
        self.calls: list[str] = []

    def __call__(self, text: str) -> list[dict[str, object]]:
        self.calls.append(text)
        return list(self.predictions)


def _prediction(
    text: str,
    value: str,
    entity: str,
    score: float,
    *,
    after: int = 0,
) -> dict[str, object]:
    start = text.index(value, after)
    return {
        "entity": entity,
        "score": score,
        "start": start,
        "end": start + len(value),
        "word": value,
    }


def test_extractor_loads_lazily_and_preserves_raw_open_value_spans() -> None:
    text = (
        "Quiero donas artesanales, sin ruido, cerca del Parque Central a 2 km."
    )
    predictions = [
        _prediction(text, "donas", "B-CATEGORY", 0.96),
        _prediction(text, "artesanales", "I-CATEGORY", 0.90),
        _prediction(text, "ruido", "B-EXCLUSION", 0.94),
        _prediction(text, "Parque", "B-LOCATION", 0.91),
        _prediction(text, "Central", "I-LOCATION", 0.89),
        _prediction(text, "2", "B-RADIUS", 0.88),
        _prediction(text, "km", "I-RADIUS", 0.86),
    ]
    classifier = FakeTokenClassifier(predictions)
    loader_calls: list[tuple[str, str | None, int | str | None]] = []

    def loader(
        model_name: str,
        revision: str | None,
        device: int | str | None,
    ) -> FakeTokenClassifier:
        loader_calls.append((model_name, revision, device))
        return classifier

    extractor = BertPlaceIntentExtractor(
        "frimeet/places-intent-bert",
        model_version="2026-07-16",
        device="cpu",
        model_loader=loader,
    )

    assert extractor.is_loaded is False
    assert loader_calls == []

    frame = extractor.extract(text)

    assert extractor.is_loaded is True
    assert loader_calls == [
        ("frimeet/places-intent-bert", "2026-07-16", "cpu")
    ]
    assert frame.raw_text == text
    assert frame.model_name == "frimeet/places-intent-bert"
    assert frame.model_version == "2026-07-16"
    assert [span.text for span in frame.spans] == [
        "donas artesanales",
        "ruido",
        "Parque Central",
        "2 km",
    ]
    assert frame.categories[0].text == "donas artesanales"
    assert frame.categories[0].polarity == "positive"
    assert frame.categories[0].token_count == 2
    assert frame.exclusions[0].text == "ruido"
    assert frame.exclusions[0].polarity == "negative"
    assert frame.by_type("LOCATION")[0].text == "Parque Central"
    assert frame.by_type("RADIUS")[0].text == "2 km"
    assert 0.86 <= frame.confidence <= 0.96
    for span in frame.spans:
        assert text[span.start : span.end] == span.text

    extractor.extract(text)
    assert len(loader_calls) == 1
    assert classifier.calls == [text, text]


def test_custom_model_labels_map_to_slots_and_contextual_polarity() -> None:
    text = "Busco cafecito con terraza pero evito ruido"
    classifier = FakeTokenClassifier(
        [
            _prediction(text, "cafecito", "B-TARGET", 0.93),
            _prediction(text, "terraza", "B-AMENITY", 0.91),
            _prediction(text, "ruido", "B-NEGATIVE_AMENITY", 0.95),
            _prediction(text, "Busco", "B-UNRELATED_HEAD", 0.99),
        ]
    )
    extractor = BertPlaceIntentExtractor(
        "injected-model",
        classifier=classifier,
        label_definitions={
            "TARGET": "CATEGORY",
            "AMENITY": ("PREFERENCE", "positive"),
            "NEGATIVE_AMENITY": ("PREFERENCE", "negative"),
        },
    )

    frame = extractor.extract(text)

    assert [span.text for span in frame.categories] == ["cafecito"]
    assert [span.text for span in frame.preferences] == ["terraza"]
    assert [span.text for span in frame.exclusions] == ["ruido"]
    assert frame.exclusions[0].slot_type == "PREFERENCE"
    assert "UNRELATED_HEAD" not in {span.slot_type for span in frame.spans}


def test_stray_i_label_is_recovered_but_a_skipped_token_breaks_the_span() -> None:
    text = "terraza muy tranquila"
    classifier = FakeTokenClassifier(
        [
            _prediction(text, "terraza", "I-PREFERENCE", 0.92),
            _prediction(text, "muy", "I-PREFERENCE", 0.30),
            _prediction(text, "tranquila", "I-PREFERENCE", 0.90),
        ]
    )
    extractor = BertPlaceIntentExtractor(
        "injected-model",
        classifier=classifier,
        minimum_token_confidence=0.8,
    )

    frame = extractor.extract(text)

    assert [span.text for span in frame.preferences] == ["terraza", "tranquila"]
    assert all(span.token_count == 1 for span in frame.preferences)


def test_huggingface_token_index_breaks_span_across_ignored_o_tokens() -> None:
    text = "cafe cerca del parque"
    first = _prediction(text, "cafe", "B-CATEGORY", 0.94)
    first["index"] = 1
    second = _prediction(text, "parque", "I-CATEGORY", 0.91)
    second["index"] = 5
    classifier = FakeTokenClassifier([first, second])

    frame = BertPlaceIntentExtractor(
        "injected-model",
        classifier=classifier,
    ).extract(text)

    assert [span.text for span in frame.categories] == ["cafe", "parque"]
    assert all(span.token_count == 1 for span in frame.categories)


def test_empty_text_returns_an_empty_frame_without_loading_the_model() -> None:
    def loader(
        _model_name: str,
        _revision: str | None,
        _device: int | str | None,
    ) -> FakeTokenClassifier:
        raise AssertionError("empty text must not load transformers")

    extractor = BertPlaceIntentExtractor("lazy-model", model_loader=loader)

    frame = extractor.extract(" \t\n")

    assert frame.raw_text == " \t\n"
    assert frame.spans == ()
    assert frame.confidence == 0.0
    assert extractor.is_loaded is False


def test_injected_classifier_does_not_require_transformers_and_accepts_group_labels() -> None:
    text = "Parque México"
    classifier = FakeTokenClassifier(
        [
            {
                "entity_group": "LOCATION",
                "score": 0.97,
                "start": 0,
                "end": len(text),
            }
        ]
    )
    extractor = BertPlaceIntentExtractor("no-transformers-needed", classifier=classifier)

    frame = extractor.extract(text)

    assert frame.by_type("LOCATION")[0].text == text
    assert frame.by_type("LOCATION")[0].confidence == pytest.approx(0.97)


def test_loader_inference_and_malformed_output_errors_have_context() -> None:
    def broken_loader(
        _model_name: str,
        _revision: str | None,
        _device: int | str | None,
    ) -> FakeTokenClassifier:
        raise OSError("model cache unavailable")

    extractor = BertPlaceIntentExtractor("missing-model", model_loader=broken_loader)
    with pytest.raises(
        BertIntentModelLoadError,
        match="missing-model.*model cache unavailable",
    ):
        extractor.extract("donas")

    class BrokenClassifier:
        def __call__(self, _text: str) -> list[dict[str, object]]:
            raise RuntimeError("backend crashed")

    extractor = BertPlaceIntentExtractor("broken-model", classifier=BrokenClassifier())
    with pytest.raises(
        BertIntentInferenceError,
        match="broken-model.*backend crashed",
    ):
        extractor.extract("donas")

    malformed = FakeTokenClassifier(
        [{"entity": "B-CATEGORY", "score": 0.9, "start": 0}]
    )
    extractor = BertPlaceIntentExtractor("bad-output", classifier=malformed)
    with pytest.raises(BertIntentOutputError, match="integer end offset"):
        extractor.extract("donas")


def test_frame_and_span_confidence_are_length_weighted() -> None:
    text = "cafe silencioso"
    classifier = FakeTokenClassifier(
        [
            _prediction(text, "cafe", "B-CATEGORY", 1.0),
            _prediction(text, "silencioso", "B-PREFERENCE", 0.5),
        ]
    )
    frame = BertPlaceIntentExtractor("model", classifier=classifier).extract(text)

    expected = (4 * 1.0 + 10 * 0.5) / 14
    assert frame.confidence == pytest.approx(expected)
    assert math.isfinite(frame.confidence)


def test_constructor_rejects_invalid_configuration() -> None:
    classifier = FakeTokenClassifier([])
    with pytest.raises(ValueError, match="model_name_or_path"):
        BertPlaceIntentExtractor(" ")
    with pytest.raises(ValueError, match="minimum_token_confidence"):
        BertPlaceIntentExtractor("model", minimum_token_confidence=1.1)
    with pytest.raises(ValueError, match="classifier or model_loader"):
        BertPlaceIntentExtractor(
            "model",
            classifier=classifier,
            model_loader=lambda _name, _revision, _device: classifier,
        )
    with pytest.raises(ValueError, match="slot_type"):
        BertPlaceIntentExtractor(
            "model",
            classifier=classifier,
            label_definitions={"CUSTOM": "NOT_A_SLOT"},  # type: ignore[dict-item]
        )
