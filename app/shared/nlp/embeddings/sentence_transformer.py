from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.embeddings.download_sentence_transformer_model import (
    ensure_sentence_transformer_model,
)
from app.shared.nlp.preprocessing.text import clean_text


class SentenceEncoderModel(Protocol):
    max_seq_length: int

    def get_sentence_embedding_dimension(self) -> int | None: ...

    def encode(self, sentences: list[str], **kwargs: Any) -> Sequence[Sequence[float]]: ...


ModelLoader = Callable[[str, str], SentenceEncoderModel]


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Asymmetric, normalized sentence embeddings for semantic retrieval."""

    def __init__(
        self,
        model_id: str,
        expected_dimension: int = 384,
        revision: str = "main",
        model_path: str | None = None,
        cache_dir: str = ".models/sentence-transformers",
        auto_download: bool = True,
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
        batch_size: int = 32,
        device: str = "cpu",
        max_sequence_length: int = 256,
        model_loader: ModelLoader | None = None,
    ) -> None:
        if model_loader is None:
            source = self._resolve_model_source(
                model_id=model_id,
                revision=revision,
                model_path=model_path,
                cache_dir=cache_dir,
                auto_download=auto_download,
            )
            model_loader = _load_sentence_transformer
        else:
            source = model_path or model_id

        self._model = model_loader(str(source), device)
        model_dimension = self._model.get_sentence_embedding_dimension()
        if model_dimension is None:
            raise ValueError("Sentence Transformer did not report an embedding dimension")
        self.dimension = int(model_dimension)
        if self.dimension != expected_dimension:
            raise ValueError(
                "Sentence Transformer dimension does not match EMBEDDING_DIMENSION: "
                f"model={self.dimension}, configured={expected_dimension}"
            )

        self._model.max_seq_length = max_sequence_length
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._batch_size = batch_size

    def embed_text(self, text: str) -> list[float]:
        return self.embed_document(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], self._query_prefix)[0]

    def embed_document(self, text: str) -> list[float]:
        return self._encode([text], self._document_prefix)[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, self._document_prefix)

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        if not texts:
            return []

        output = [[0.0] * self.dimension for _ in texts]
        nonempty: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            cleaned = clean_text(text)
            if cleaned:
                nonempty.append((index, f"{prefix}{cleaned}"))
        if not nonempty:
            return output

        encoded = self._model.encode(
            [value for _, value in nonempty],
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if len(encoded) != len(nonempty):
            raise ValueError("Sentence Transformer returned an unexpected batch size")

        for (output_index, _), vector in zip(nonempty, encoded):
            values = [float(value) for value in vector]
            if len(values) != self.dimension:
                raise ValueError(
                    "Sentence Transformer returned dimension "
                    f"{len(values)}; expected {self.dimension}"
                )
            output[output_index] = values
        return output

    @staticmethod
    def _resolve_model_source(
        model_id: str,
        revision: str,
        model_path: str | None,
        cache_dir: str,
        auto_download: bool,
    ) -> Path:
        if model_path:
            path = Path(model_path).expanduser()
            if path.is_dir() and (path / "modules.json").is_file():
                return path
            raise FileNotFoundError(
                f"SENTENCE_TRANSFORMER_MODEL_PATH is not a valid model: {path}"
            )
        return ensure_sentence_transformer_model(
            cache_dir=cache_dir,
            repo_id=model_id,
            revision=revision,
            auto_download=auto_download,
        )


def _load_sentence_transformer(path: str, device: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for contextual embeddings"
        ) from exc
    return SentenceTransformer(path, device=device, local_files_only=True)
