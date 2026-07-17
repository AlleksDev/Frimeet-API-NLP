"""Lazy BERT token-classification adapter for open place-chat slots.

The extractor deliberately returns raw concepts instead of canonical place
categories.  Category alignment belongs to the semantic catalog/retrieval
stage; doing it here would recreate a closed taxonomy in the model adapter.

The default loader imports ``transformers`` only on the first non-empty call.
Tests and alternative serving runtimes can inject a callable classifier or a
loader, so importing this module never requires the optional ML dependency.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, cast


SlotType = Literal[
    "CATEGORY",
    "PREFERENCE",
    "EXCLUSION",
    "LOCATION",
    "REFERENCE",
    "RADIUS",
]
SpanPolarity = Literal["positive", "negative", "neutral"]
IOBPrefix = Literal["B", "I"]

_SLOT_TYPES: frozenset[str] = frozenset(
    {
        "CATEGORY",
        "PREFERENCE",
        "EXCLUSION",
        "LOCATION",
        "REFERENCE",
        "RADIUS",
    }
)
_POLARITIES: frozenset[str] = frozenset({"positive", "negative", "neutral"})


class BertIntentExtractionError(RuntimeError):
    """Base error raised by the contextual intent extractor."""


class BertIntentModelLoadError(BertIntentExtractionError):
    """The configured token-classification model could not be loaded."""


class BertIntentInferenceError(BertIntentExtractionError):
    """The loaded model failed while processing a message."""


class BertIntentOutputError(BertIntentExtractionError):
    """The model returned an invalid token-classification payload."""


@dataclass(frozen=True)
class SlotLabelDefinition:
    """Maps one model label to a domain slot and its semantic polarity.

    Mapping is intentionally configuration-driven.  For example, a model can
    expose ``B-AMENITY`` and map ``AMENITY`` to a positive ``PREFERENCE``, or
    expose ``B-NEGATIVE_AMENITY`` and map it to a negative ``PREFERENCE``.
    """

    slot_type: SlotType
    polarity: SpanPolarity = "neutral"

    def __post_init__(self) -> None:
        slot_type = str(self.slot_type).strip().upper()
        polarity = str(self.polarity).strip().lower()
        if slot_type not in _SLOT_TYPES:
            raise ValueError(
                "slot_type must be one of " + ", ".join(sorted(_SLOT_TYPES))
            )
        if polarity not in _POLARITIES:
            raise ValueError(
                "polarity must be one of " + ", ".join(sorted(_POLARITIES))
            )
        object.__setattr__(self, "slot_type", cast(SlotType, slot_type))
        object.__setattr__(self, "polarity", cast(SpanPolarity, polarity))


@dataclass(frozen=True)
class IntentSpan:
    """One contextual slot using character offsets into the original message."""

    slot_type: SlotType
    text: str
    start: int
    end: int
    polarity: SpanPolarity
    confidence: float
    token_count: int = 1

    def __post_init__(self) -> None:
        if self.slot_type not in _SLOT_TYPES:
            raise ValueError(f"unsupported slot_type: {self.slot_type!r}")
        if self.polarity not in _POLARITIES:
            raise ValueError(f"unsupported polarity: {self.polarity!r}")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("span offsets must satisfy 0 <= start < end")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("span text must not be empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("span confidence must be finite and between 0 and 1")
        if self.token_count <= 0:
            raise ValueError("span token_count must be positive")


@dataclass(frozen=True)
class IntentFrame:
    """Open contextual interpretation of one unmodified user message."""

    raw_text: str
    spans: tuple[IntentSpan, ...]
    confidence: float
    model_name: str
    model_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.raw_text, str):
            raise TypeError("raw_text must be str")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("frame confidence must be finite and between 0 and 1")
        for span in self.spans:
            if span.end > len(self.raw_text):
                raise ValueError("span extends beyond raw_text")
            if self.raw_text[span.start : span.end] != span.text:
                raise ValueError("span text must match raw_text character offsets")

    def by_type(self, slot_type: SlotType) -> tuple[IntentSpan, ...]:
        """Return all spans of a type without normalizing their raw values."""

        return tuple(span for span in self.spans if span.slot_type == slot_type)

    @property
    def categories(self) -> tuple[IntentSpan, ...]:
        return self.by_type("CATEGORY")

    @property
    def preferences(self) -> tuple[IntentSpan, ...]:
        return tuple(
            span
            for span in self.spans
            if span.slot_type == "PREFERENCE" and span.polarity == "positive"
        )

    @property
    def exclusions(self) -> tuple[IntentSpan, ...]:
        return tuple(
            span
            for span in self.spans
            if span.slot_type == "EXCLUSION" or span.polarity == "negative"
        )


class TokenClassifier(Protocol):
    """Minimal shape shared by a Hugging Face pipeline and test doubles."""

    def __call__(self, text: str) -> Sequence[Mapping[str, Any]]:
        """Return token rows containing an IOB label, score, start and end."""


TokenClassifierLoader: TypeAlias = Callable[
    [str, str | None, int | str | None], TokenClassifier
]
LabelDefinitionInput: TypeAlias = (
    SlotLabelDefinition | SlotType | tuple[SlotType, SpanPolarity]
)


DEFAULT_LABEL_DEFINITIONS: Mapping[str, SlotLabelDefinition] = {
    "CATEGORY": SlotLabelDefinition("CATEGORY", "positive"),
    "PREFERENCE": SlotLabelDefinition("PREFERENCE", "positive"),
    "EXCLUSION": SlotLabelDefinition("EXCLUSION", "negative"),
    "LOCATION": SlotLabelDefinition("LOCATION", "neutral"),
    "REFERENCE": SlotLabelDefinition("REFERENCE", "neutral"),
    "RADIUS": SlotLabelDefinition("RADIUS", "neutral"),
}


@dataclass(frozen=True)
class _TokenPrediction:
    prefix: IOBPrefix
    definition: SlotLabelDefinition
    start: int
    end: int
    score: float
    order: int


@dataclass
class _SpanBuilder:
    definition: SlotLabelDefinition
    start: int
    end: int
    weighted_score: float
    score_weight: int
    token_count: int
    last_order: int

    @classmethod
    def from_token(cls, token: _TokenPrediction) -> _SpanBuilder:
        weight = max(1, token.end - token.start)
        return cls(
            definition=token.definition,
            start=token.start,
            end=token.end,
            weighted_score=token.score * weight,
            score_weight=weight,
            token_count=1,
            last_order=token.order,
        )

    def append(self, token: _TokenPrediction) -> None:
        weight = max(1, token.end - token.start)
        self.end = max(self.end, token.end)
        self.weighted_score += token.score * weight
        self.score_weight += weight
        self.token_count += 1
        self.last_order = token.order

    def build(self, raw_text: str) -> IntentSpan:
        return IntentSpan(
            slot_type=self.definition.slot_type,
            text=raw_text[self.start : self.end],
            start=self.start,
            end=self.end,
            polarity=self.definition.polarity,
            confidence=self.weighted_score / self.score_weight,
            token_count=self.token_count,
        )


class BertPlaceIntentExtractor:
    """Extract contextual, open-value slots with a fine-tuned BERT model.

    ``label_definitions`` is merged over the standard six slot labels.  Keys
    refer to model labels *without* the IOB prefix.  Unknown labels are ignored,
    which lets one model expose additional tasks without coupling Places to
    them.  Values can be ``SlotLabelDefinition``, a canonical slot string, or a
    ``(slot, polarity)`` tuple.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        model_version: str | None = None,
        device: int | str | None = None,
        label_definitions: Mapping[str, LabelDefinitionInput] | None = None,
        minimum_token_confidence: float = 0.0,
        classifier: TokenClassifier | None = None,
        model_loader: TokenClassifierLoader | None = None,
    ) -> None:
        model_name = _required_text(model_name_or_path, "model_name_or_path")
        if not math.isfinite(minimum_token_confidence) or not (
            0.0 <= minimum_token_confidence <= 1.0
        ):
            raise ValueError(
                "minimum_token_confidence must be finite and between 0 and 1"
            )
        if classifier is not None and model_loader is not None:
            raise ValueError("provide classifier or model_loader, not both")
        if classifier is not None and not callable(classifier):
            raise TypeError("classifier must be callable")

        definitions = dict(DEFAULT_LABEL_DEFINITIONS)
        for raw_label, raw_definition in (label_definitions or {}).items():
            label = _normalized_model_label(raw_label)
            definitions[label] = _coerce_label_definition(raw_definition)

        self._model_name = model_name
        self._model_version = (
            _required_text(model_version, "model_version")
            if model_version is not None
            else "unspecified"
        )
        self._revision = model_version
        self._device = device
        self._definitions = definitions
        self._minimum_token_confidence = minimum_token_confidence
        self._classifier = classifier
        self._model_loader = model_loader or _load_transformers_classifier
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._classifier is not None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    def extract(self, text: str) -> IntentFrame:
        """Extract an immutable frame while preserving ``text`` byte-for-byte."""

        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not text.strip():
            return IntentFrame(
                raw_text=text,
                spans=(),
                confidence=0.0,
                model_name=self._model_name,
                model_version=self._model_version,
            )

        classifier = self._ensure_classifier()
        try:
            # Hugging Face pipelines and GPU modules are not guaranteed to be
            # safe under concurrent calls. Parsing already runs in a worker
            # thread, so serializing one model does not block the event loop.
            with self._inference_lock:
                raw_predictions = classifier(text)
        except Exception as exc:
            raise BertIntentInferenceError(
                f"token classification failed for {self._model_name!r}: {exc}"
            ) from exc

        predictions = self._prepare_predictions(raw_predictions, text)
        spans = _decode_iob(predictions, text)
        return IntentFrame(
            raw_text=text,
            spans=spans,
            confidence=_frame_confidence(spans),
            model_name=self._model_name,
            model_version=self._model_version,
        )

    def _ensure_classifier(self) -> TokenClassifier:
        classifier = self._classifier
        if classifier is not None:
            return classifier

        with self._load_lock:
            classifier = self._classifier
            if classifier is not None:
                return classifier
            try:
                classifier = self._model_loader(
                    self._model_name,
                    self._revision,
                    self._device,
                )
            except BertIntentModelLoadError:
                raise
            except Exception as exc:
                raise BertIntentModelLoadError(
                    f"could not load token-classification model "
                    f"{self._model_name!r}: {exc}"
                ) from exc
            if not callable(classifier):
                raise BertIntentModelLoadError(
                    f"loader for {self._model_name!r} did not return a callable"
                )
            self._classifier = classifier
            return classifier

    def _prepare_predictions(
        self,
        raw_predictions: Sequence[Mapping[str, Any]],
        text: str,
    ) -> tuple[_TokenPrediction, ...]:
        if isinstance(raw_predictions, (str, bytes, Mapping)) or not isinstance(
            raw_predictions, Sequence
        ):
            raise BertIntentOutputError(
                "token classifier output must be a sequence of mappings"
            )

        predictions: list[_TokenPrediction] = []
        for order, row in enumerate(raw_predictions):
            if not isinstance(row, Mapping):
                raise BertIntentOutputError(
                    f"token classifier row {order} must be a mapping"
                )
            label_value = row.get("entity", row.get("entity_group", row.get("label")))
            if not isinstance(label_value, str):
                raise BertIntentOutputError(
                    f"token classifier row {order} has no string entity label"
                )
            prefix, base_label = _split_iob_label(label_value)
            if base_label is None:
                continue
            definition = self._definitions.get(base_label)
            if definition is None:
                continue

            score = _finite_score(row.get("score"), order)
            if score < self._minimum_token_confidence:
                continue
            start = _integer_offset(row.get("start"), "start", order)
            end = _integer_offset(row.get("end"), "end", order)
            if start < 0 or end <= start or end > len(text):
                raise BertIntentOutputError(
                    f"token classifier row {order} has invalid offsets "
                    f"start={start}, end={end}, text_length={len(text)}"
                )
            token_order = _token_order(row.get("index"), order)
            predictions.append(
                _TokenPrediction(
                    prefix=prefix,
                    definition=definition,
                    start=start,
                    end=end,
                    score=score,
                    order=token_order,
                )
            )
        return tuple(
            sorted(
                predictions,
                key=lambda token: (token.start, token.end, token.order),
            )
        )


def _decode_iob(
    predictions: Sequence[_TokenPrediction],
    raw_text: str,
) -> tuple[IntentSpan, ...]:
    spans: list[IntentSpan] = []
    current: _SpanBuilder | None = None

    for token in predictions:
        continues_current = bool(
            token.prefix == "I"
            and current is not None
            and current.definition == token.definition
            and token.start >= current.end
            and token.order == current.last_order + 1
        )
        if continues_current:
            assert current is not None
            current.append(token)
            continue

        if current is not None:
            spans.append(current.build(raw_text))
        # A stray I-label begins a recoverable new span instead of losing the
        # user's concept because of one imperfect model transition.
        current = _SpanBuilder.from_token(token)

    if current is not None:
        spans.append(current.build(raw_text))
    return tuple(spans)


def _frame_confidence(spans: Sequence[IntentSpan]) -> float:
    if not spans:
        return 0.0
    weights = [max(1, span.end - span.start) for span in spans]
    return sum(
        span.confidence * weight for span, weight in zip(spans, weights)
    ) / sum(weights)


def _split_iob_label(raw_label: str) -> tuple[IOBPrefix, str | None]:
    label = raw_label.strip().upper()
    if not label:
        raise BertIntentOutputError("token classifier returned an empty label")
    if label == "O":
        return "B", None
    if len(label) > 2 and label[0] in {"B", "I"} and label[1] in {"-", "_"}:
        return cast(IOBPrefix, label[0]), _normalized_model_label(label[2:])
    # Aggregated Hugging Face pipelines may return CATEGORY instead of
    # B-CATEGORY.  Treat an unprefixed label as one complete span.
    return "B", _normalized_model_label(label)


def _coerce_label_definition(value: LabelDefinitionInput) -> SlotLabelDefinition:
    if isinstance(value, SlotLabelDefinition):
        return value
    if isinstance(value, str):
        slot_type = value.strip().upper()
        default_definition = DEFAULT_LABEL_DEFINITIONS.get(slot_type)
        if default_definition is not None:
            return default_definition
        return SlotLabelDefinition(cast(SlotType, slot_type))
    if isinstance(value, tuple) and len(value) == 2:
        return SlotLabelDefinition(
            cast(SlotType, value[0]),
            cast(SpanPolarity, value[1]),
        )
    raise TypeError(
        "label definition must be SlotLabelDefinition, a slot string, or a "
        "(slot, polarity) tuple"
    )


def _normalized_model_label(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"model label must be str, got {type(value).__name__}")
    label = value.strip().upper()
    if not label:
        raise ValueError("model label must not be empty")
    return label


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str, got {type(value).__name__}")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _finite_score(value: Any, row: int) -> float:
    if isinstance(value, bool):
        raise BertIntentOutputError(f"token classifier row {row} has invalid score")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise BertIntentOutputError(
            f"token classifier row {row} has no numeric score"
        ) from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise BertIntentOutputError(
            f"token classifier row {row} score must be finite and between 0 and 1"
        )
    return score


def _integer_offset(value: Any, name: str, row: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BertIntentOutputError(
            f"token classifier row {row} has no integer {name} offset"
        )
    return value


def _token_order(value: Any, fallback: int) -> int:
    """Preserve Hugging Face token adjacency after ignored labels are removed."""

    if value is None:
        return fallback
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BertIntentOutputError(
            "token classifier output has an invalid token index"
        )
    return value


def _load_transformers_classifier(
    model_name_or_path: str,
    revision: str | None,
    device: int | str | None,
) -> TokenClassifier:
    """Load the optional Hugging Face implementation on first inference."""

    try:
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )
    except ImportError as exc:
        raise BertIntentModelLoadError(
            "transformers is required to load the BERT intent extractor; "
            "install it or inject classifier/model_loader"
        ) from exc

    pretrained_kwargs: dict[str, Any] = {"trust_remote_code": False}
    if revision is not None:
        pretrained_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        **pretrained_kwargs,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        **pretrained_kwargs,
    )
    pipeline_kwargs: dict[str, Any] = {
        "task": "token-classification",
        "model": model,
        "tokenizer": tokenizer,
        "aggregation_strategy": "none",
        "ignore_labels": ["O"],
    }
    if device is not None:
        pipeline_kwargs["device"] = device
    return cast(TokenClassifier, pipeline(**pipeline_kwargs))
