from __future__ import annotations

import pytest

from app.shared.nlp.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)


class FakeSentenceEncoder:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.max_seq_length = 512
        self.calls: list[list[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, sentences: list[str], **kwargs) -> list[list[float]]:
        self.calls.append(sentences)
        assert kwargs["normalize_embeddings"] is True
        return [[1.0] + [0.0] * (self.dimension - 1) for _ in sentences]


def test_sentence_transformer_uses_asymmetric_prefixes_and_original_text() -> None:
    fake_model = FakeSentenceEncoder()
    provider = SentenceTransformerEmbeddingProvider(
        model_id="fake/e5",
        expected_dimension=3,
        query_prefix="query: ",
        document_prefix="passage: ",
        max_sequence_length=128,
        model_loader=lambda _path, _device: fake_model,
    )

    provider.embed_query("Cafetería cerca del parque")
    provider.embed_document("Cafetería con postres")

    assert fake_model.calls == [
        ["query: Cafetería cerca del parque"],
        ["passage: Cafetería con postres"],
    ]
    assert fake_model.max_seq_length == 128


def test_sentence_transformer_batches_documents_and_keeps_empty_as_zero() -> None:
    fake_model = FakeSentenceEncoder()
    provider = SentenceTransformerEmbeddingProvider(
        model_id="fake/e5",
        expected_dimension=3,
        model_loader=lambda _path, _device: fake_model,
    )

    embeddings = provider.embed_documents(["museo de arte", "", "parque natural"])

    assert embeddings == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert fake_model.calls == [["passage: museo de arte", "passage: parque natural"]]


def test_sentence_transformer_rejects_mismatched_dimension() -> None:
    with pytest.raises(ValueError, match="does not match"):
        SentenceTransformerEmbeddingProvider(
            model_id="fake/e5",
            expected_dimension=384,
            model_loader=lambda _path, _device: FakeSentenceEncoder(dimension=3),
        )
