"""Lazy Sentence-Transformer embeddings with strict output validation.

The optional ``sentence-transformers`` dependency is intentionally imported only
when the first non-empty text is embedded.  This keeps application startup and
test discovery independent from heavyweight model loading.
"""

from __future__ import annotations

import inspect
import math
import threading
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Protocol

from app.shared.nlp.embeddings.base import EmbeddingProvider


class SentenceTransformerModel(Protocol):
    """Small portion of the SentenceTransformer API used by this provider."""

    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...

    def get_sentence_embedding_dimension(self) -> int | None: ...


ModelLoader = Callable[[str, str | None], SentenceTransformerModel]


_SHARED_MODELS: dict[
    tuple[str, str | None, str | None, bool], SentenceTransformerModel
] = {}
_SHARED_MODELS_LOCK = threading.Lock()
_MODEL_INFERENCE_LOCKS: dict[int, threading.Lock] = {}


class SentenceTransformerEmbeddingError(RuntimeError):
    """Base error raised when a sentence embedding cannot be produced safely."""


class SentenceTransformerModelLoadError(SentenceTransformerEmbeddingError):
    """Raised when the configured model or its runtime cannot be loaded."""


class SentenceTransformerInferenceError(SentenceTransformerEmbeddingError):
    """Raised when model inference fails."""


class SentenceTransformerDimensionError(ValueError):
    """Raised when a model returns a vector with an unexpected dimension."""


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Batch Sentence-Transformer provider with lazy, thread-safe model loading.

    ``text_prefix`` supports retrieval models such as E5, whose query and
    passage encoders use the same weights but require different input prefixes.
    Configure separate provider instances for queries and documents when those
    prefixes differ.
    """

    def __init__(
        self,
        model_name_or_path: str,
        expected_dimension: int,
        *,
        batch_size: int = 32,
        device: str | None = None,
        model_revision: str | None = None,
        fix_mistral_regex: bool = True,
        text_prefix: str = "",
        normalize_embeddings: bool = True,
        model_loader: ModelLoader | None = None,
    ) -> None:
        if not model_name_or_path.strip():
            raise ValueError("model_name_or_path must not be empty")
        if expected_dimension <= 0:
            raise ValueError("expected_dimension must be greater than zero")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self.model_name_or_path = model_name_or_path
        self.dimension = int(expected_dimension)
        self.batch_size = int(batch_size)
        self.device = device
        self.model_revision = (
            model_revision.strip() or None if model_revision is not None else None
        )
        self.fix_mistral_regex = bool(fix_mistral_regex)
        self.text_prefix = text_prefix
        self.normalize_embeddings = normalize_embeddings

        # Keep injected two-argument loaders backward-compatible while binding
        # the immutable Hub revision only for the default loader.
        self._model_loader = model_loader or partial(
            _load_sentence_transformer,
            revision=self.model_revision,
            fix_mistral_regex=self.fix_mistral_regex,
        )
        self._model: SentenceTransformerModel | None = None
        self._model_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """Whether the heavyweight model has already been initialized."""

        return self._model is not None

    def embed_text(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        prepared: list[str] = []
        positions: list[int] = []
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"texts[{index}] must be str, got {type(text).__name__}"
                )
            stripped = text.strip()
            if stripped:
                positions.append(index)
                prepared.append(f"{self.text_prefix}{stripped}")

        embeddings = [[0.0] * self.dimension for _ in texts]
        if not prepared:
            return embeddings

        model = self._get_model()
        try:
            with _inference_lock_for(model):
                raw_embeddings = model.encode(
                    prepared,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
        except Exception as exc:
            raise SentenceTransformerInferenceError(
                "Sentence-Transformer inference failed for "
                f"model {self.model_name_or_path!r}: {exc}"
            ) from exc

        vectors = self._validate_and_convert(raw_embeddings, len(prepared))
        for position, vector in zip(positions, vectors):
            embeddings[position] = vector
        return embeddings

    def _get_model(self) -> SentenceTransformerModel:
        model = self._model
        if model is not None:
            return model

        with self._model_lock:
            model = self._model
            if model is not None:
                return model
            try:
                model = self._model_loader(self.model_name_or_path, self.device)
            except SentenceTransformerEmbeddingError:
                raise
            except Exception as exc:
                raise SentenceTransformerModelLoadError(
                    "Could not load Sentence-Transformer model "
                    f"{self.model_name_or_path!r}: {exc}"
                ) from exc

            self._validate_model_dimension(model)
            self._model = model
            return model

    def _validate_model_dimension(self, model: SentenceTransformerModel) -> None:
        dimension_getter = getattr(
            model, "get_sentence_embedding_dimension", None
        )
        if not callable(dimension_getter):
            return
        try:
            reported_dimension = dimension_getter()
        except Exception as exc:
            raise SentenceTransformerModelLoadError(
                "Could not inspect the embedding dimension for "
                f"model {self.model_name_or_path!r}: {exc}"
            ) from exc
        if reported_dimension is None:
            return
        if int(reported_dimension) != self.dimension:
            raise SentenceTransformerDimensionError(
                "Sentence-Transformer model dimension does not match the "
                "configured embedding dimension: "
                f"model={reported_dimension}, configured={self.dimension}, "
                f"name={self.model_name_or_path!r}"
            )

    def _validate_and_convert(
        self,
        raw_embeddings: Any,
        expected_count: int,
    ) -> list[list[float]]:
        converted = (
            raw_embeddings.tolist()
            if hasattr(raw_embeddings, "tolist")
            else raw_embeddings
        )
        try:
            rows = list(converted)
        except (TypeError, ValueError) as exc:
            raise SentenceTransformerInferenceError(
                "Sentence-Transformer returned a non-iterable embedding result"
            ) from exc

        # Some compatible runtimes collapse a one-item batch to one dimension.
        if expected_count == 1 and rows and _is_scalar(rows[0]):
            rows = [rows]
        if len(rows) != expected_count:
            raise SentenceTransformerInferenceError(
                "Sentence-Transformer returned an unexpected number of vectors: "
                f"returned={len(rows)}, expected={expected_count}"
            )

        vectors: list[list[float]] = []
        for row_index, row in enumerate(rows):
            values_source = row.tolist() if hasattr(row, "tolist") else row
            try:
                values = [float(value) for value in values_source]
            except (TypeError, ValueError) as exc:
                raise SentenceTransformerInferenceError(
                    "Sentence-Transformer returned a non-numeric vector at "
                    f"batch index {row_index}"
                ) from exc
            if len(values) != self.dimension:
                raise SentenceTransformerDimensionError(
                    "Sentence-Transformer returned an unexpected vector "
                    f"dimension at batch index {row_index}: "
                    f"returned={len(values)}, expected={self.dimension}"
                )
            if not all(math.isfinite(value) for value in values):
                raise SentenceTransformerInferenceError(
                    "Sentence-Transformer returned NaN or infinity at "
                    f"batch index {row_index}"
                )
            if self.normalize_embeddings:
                values = _l2_normalize_nonzero(values, row_index)
            vectors.append(values)
        return vectors


def _load_sentence_transformer(
    model_name_or_path: str,
    device: str | None,
    *,
    revision: str | None = None,
    fix_mistral_regex: bool = True,
) -> SentenceTransformerModel:
    cache_key = (model_name_or_path, revision, device, fix_mistral_regex)
    cached = _SHARED_MODELS.get(cache_key)
    if cached is not None:
        return cached
    with _SHARED_MODELS_LOCK:
        cached = _SHARED_MODELS.get(cache_key)
        if cached is not None:
            return cached
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SentenceTransformerModelLoadError(
                "sentence-transformers is required for BERT embeddings. "
                "Install the project's optional Sentence-Transformer dependencies."
            ) from exc

        try:
            tokenizer_kwargs = {
                "fix_mistral_regex": fix_mistral_regex,
            }
            try:
                constructor_parameters = inspect.signature(
                    SentenceTransformer
                ).parameters
            except (TypeError, ValueError):
                constructor_parameters = {}
            tokenizer_parameter = (
                "processor_kwargs"
                if "processor_kwargs" in constructor_parameters
                else "tokenizer_kwargs"
            )
            model = SentenceTransformer(
                model_name_or_path,
                device=device,
                revision=revision,
                **{tokenizer_parameter: tokenizer_kwargs},
            )
        except Exception as exc:
            raise SentenceTransformerModelLoadError(
                "Could not initialize Sentence-Transformer model "
                f"{model_name_or_path!r}: {exc}"
            ) from exc
        _SHARED_MODELS[cache_key] = model
        return model


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, complex))


def _inference_lock_for(model: SentenceTransformerModel) -> threading.Lock:
    model_id = id(model)
    with _SHARED_MODELS_LOCK:
        return _MODEL_INFERENCE_LOCKS.setdefault(model_id, threading.Lock())


def _l2_normalize_nonzero(
    vector: list[float],
    row_index: int,
) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise SentenceTransformerInferenceError(
            "Sentence-Transformer returned a zero-norm vector for non-empty "
            f"text at batch index {row_index}"
        )
    return [value / norm for value in vector]
