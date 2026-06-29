import math

import pytest

from app.shared.nlp.embeddings.fasttext import FastTextEmbeddingProvider
from app.shared.nlp.embeddings.versioning import versioned_embedding_hash


class FakeFastTextModel:
    vectors = {
        "agua": [1.0, 0.0, 0.0],
        "problemas": [0.0, 1.0, 0.0],
        "hidrico": [0.8, 0.2, 0.0],
    }

    def get_dimension(self) -> int:
        return 3

    def get_word_vector(self, word: str) -> list[float]:
        return self.vectors.get(word, [0.0, 0.0, 1.0])


def test_fasttext_provider_mean_pools_and_normalizes_tokens() -> None:
    provider = FastTextEmbeddingProvider(
        model_path="unused.bin",
        expected_dimension=3,
        model_loader=lambda _: FakeFastTextModel(),
    )

    embedding = provider.embed_text("problemas de agua")

    expected = 1 / math.sqrt(2)
    assert embedding == pytest.approx([expected, expected, 0.0])
    assert sum(value * value for value in embedding) == pytest.approx(1.0)


def test_fasttext_provider_rejects_mismatched_dimension() -> None:
    with pytest.raises(ValueError, match="does not match"):
        FastTextEmbeddingProvider(
            model_path="unused.bin",
            expected_dimension=300,
            model_loader=lambda _: FakeFastTextModel(),
        )


def test_embedding_hash_changes_with_model_configuration() -> None:
    first = versioned_embedding_hash("source", "fasttext", "v1", 300)
    second = versioned_embedding_hash("source", "fasttext", "v2", 300)

    assert first != second
