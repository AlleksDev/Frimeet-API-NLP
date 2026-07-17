import math

import pytest

from app.shared.nlp.embeddings.sentence_transformer import (
    SentenceTransformerDimensionError,
    SentenceTransformerEmbeddingProvider,
    SentenceTransformerInferenceError,
    SentenceTransformerModelLoadError,
)


class FakeSentenceTransformer:
    def __init__(
        self,
        vectors: dict[str, list[float]],
        dimension: int = 3,
    ) -> None:
        self.vectors = vectors
        self.dimension = dimension
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, sentences: list[str], **kwargs: object) -> list[list[float]]:
        self.calls.append((list(sentences), kwargs))
        return [self.vectors[text] for text in sentences]


def test_provider_loads_lazily_and_batches_non_empty_texts() -> None:
    model = FakeSentenceTransformer(
        {
            "query: donas": [3.0, 4.0, 0.0],
            "query: cafecito": [0.0, 0.0, 2.0],
        }
    )
    loader_calls: list[tuple[str, str | None]] = []

    def loader(name: str, device: str | None) -> FakeSentenceTransformer:
        loader_calls.append((name, device))
        return model

    provider = SentenceTransformerEmbeddingProvider(
        "domain-model",
        expected_dimension=3,
        batch_size=8,
        device="cpu",
        text_prefix="query: ",
        model_loader=loader,
    )

    assert provider.is_loaded is False
    assert provider.dimension == 3
    assert loader_calls == []

    embeddings = provider.embed_batch([" donas ", "  ", "cafecito"])

    assert provider.is_loaded is True
    assert loader_calls == [("domain-model", "cpu")]
    assert embeddings[0] == pytest.approx([0.6, 0.8, 0.0])
    assert embeddings[1] == [0.0, 0.0, 0.0]
    assert embeddings[2] == pytest.approx([0.0, 0.0, 1.0])
    assert model.calls == [
        (
            ["query: donas", "query: cafecito"],
            {
                "batch_size": 8,
                "convert_to_numpy": True,
                "normalize_embeddings": False,
                "show_progress_bar": False,
            },
        )
    ]


def test_empty_text_returns_zero_without_loading_model() -> None:
    def loader(
        _name: str, _device: str | None
    ) -> FakeSentenceTransformer:
        raise AssertionError("empty input must not load the model")

    provider = SentenceTransformerEmbeddingProvider(
        "domain-model",
        expected_dimension=3,
        model_loader=loader,
    )

    assert provider.embed_text("\t \n") == [0.0, 0.0, 0.0]
    assert provider.embed_batch([]) == []
    assert provider.is_loaded is False


def test_model_is_loaded_only_once_across_calls() -> None:
    model = FakeSentenceTransformer({"first": [1.0, 0.0, 0.0], "second": [0.0, 1.0, 0.0]})
    load_count = 0

    def loader(
        _name: str, _device: str | None
    ) -> FakeSentenceTransformer:
        nonlocal load_count
        load_count += 1
        return model

    provider = SentenceTransformerEmbeddingProvider(
        "domain-model",
        expected_dimension=3,
        model_loader=loader,
    )

    provider.embed_text("first")
    provider.embed_text("second")

    assert load_count == 1
    assert len(model.calls) == 2


def test_provider_rejects_reported_model_dimension_mismatch() -> None:
    model = FakeSentenceTransformer({}, dimension=4)
    provider = SentenceTransformerEmbeddingProvider(
        "wrong-model",
        expected_dimension=3,
        model_loader=lambda _name, _device: model,
    )

    with pytest.raises(
        SentenceTransformerDimensionError,
        match="model=4, configured=3",
    ):
        provider.embed_text("donas")

    assert provider.is_loaded is False


def test_provider_rejects_bad_inference_output() -> None:
    model = FakeSentenceTransformer({"donas": [1.0, 2.0]}, dimension=3)
    provider = SentenceTransformerEmbeddingProvider(
        "bad-output-model",
        expected_dimension=3,
        model_loader=lambda _name, _device: model,
    )

    with pytest.raises(
        SentenceTransformerDimensionError,
        match="returned=2, expected=3",
    ):
        provider.embed_text("donas")


def test_provider_wraps_loader_and_inference_failures_with_context() -> None:
    def broken_loader(_name: str, _device: str | None) -> FakeSentenceTransformer:
        raise OSError("model cache unavailable")

    provider = SentenceTransformerEmbeddingProvider(
        "missing-model",
        expected_dimension=3,
        model_loader=broken_loader,
    )
    with pytest.raises(
        SentenceTransformerModelLoadError,
        match="missing-model.*model cache unavailable",
    ):
        provider.embed_text("donas")

    class BrokenModel(FakeSentenceTransformer):
        def encode(
            self, sentences: list[str], **kwargs: object
        ) -> list[list[float]]:
            raise RuntimeError("backend crashed")

    broken_model = BrokenModel({}, dimension=3)
    provider = SentenceTransformerEmbeddingProvider(
        "broken-model",
        expected_dimension=3,
        model_loader=lambda _name, _device: broken_model,
    )
    with pytest.raises(
        SentenceTransformerInferenceError,
        match="broken-model.*backend crashed",
    ):
        provider.embed_text("donas")


def test_provider_rejects_zero_norm_and_non_finite_vectors() -> None:
    zero_model = FakeSentenceTransformer({"zero": [0.0, 0.0, 0.0]})
    provider = SentenceTransformerEmbeddingProvider(
        "zero-model",
        expected_dimension=3,
        model_loader=lambda _name, _device: zero_model,
    )
    with pytest.raises(SentenceTransformerInferenceError, match="zero-norm"):
        provider.embed_text("zero")

    nan_model = FakeSentenceTransformer({"nan": [math.nan, 0.0, 1.0]})
    provider = SentenceTransformerEmbeddingProvider(
        "nan-model",
        expected_dimension=3,
        model_loader=lambda _name, _device: nan_model,
    )
    with pytest.raises(
        SentenceTransformerInferenceError,
        match="NaN or infinity",
    ):
        provider.embed_text("nan")


def test_provider_validates_constructor_and_input() -> None:
    with pytest.raises(ValueError, match="model_name_or_path"):
        SentenceTransformerEmbeddingProvider(" ", expected_dimension=3)
    with pytest.raises(ValueError, match="expected_dimension"):
        SentenceTransformerEmbeddingProvider("model", expected_dimension=0)
    with pytest.raises(ValueError, match="batch_size"):
        SentenceTransformerEmbeddingProvider(
            "model", expected_dimension=3, batch_size=0
        )

    provider = SentenceTransformerEmbeddingProvider(
        "model",
        expected_dimension=3,
        model_loader=lambda _name, _device: FakeSentenceTransformer({}),
    )
    with pytest.raises(TypeError, match=r"texts\[1\].*int"):
        provider.embed_batch(["valid", 42])  # type: ignore[list-item]
