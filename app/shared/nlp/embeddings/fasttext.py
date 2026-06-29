import math
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from app.shared.nlp.embeddings.base import EmbeddingProvider
from app.shared.nlp.embeddings.download_fasttext_model import ensure_fasttext_model
from app.shared.nlp.preprocessing.text import tokenize_for_embeddings


class FastTextModel(Protocol):
    def get_dimension(self) -> int: ...

    def get_word_vector(self, word: str) -> Sequence[float]: ...


ModelLoader = Callable[[str], FastTextModel]


class FastTextEmbeddingProvider(EmbeddingProvider):
    """Mean-pooled, L2-normalized Spanish FastText document embeddings."""

    def __init__(
        self,
        model_path: str,
        expected_dimension: int = 300,
        repo_id: str = "facebook/fasttext-es-vectors",
        filename: str = "model.bin",
        auto_download: bool = True,
        model_loader: ModelLoader | None = None,
    ) -> None:
        if model_loader is None:
            resolved_path = self._resolve_model_path(
                model_path=model_path,
                repo_id=repo_id,
                filename=filename,
                auto_download=auto_download,
            )
            model_loader = _load_fasttext_model
        else:
            resolved_path = Path(model_path)

        self._model = model_loader(str(resolved_path))
        self.dimension = int(self._model.get_dimension())
        if self.dimension != expected_dimension:
            raise ValueError(
                "FastText model dimension does not match EMBEDDING_DIMENSION: "
                f"model={self.dimension}, configured={expected_dimension}"
            )

    def embed_text(self, text: str) -> list[float]:
        tokens = tokenize_for_embeddings(text)
        if not tokens:
            return [0.0] * self.dimension

        summed = [0.0] * self.dimension
        for token in tokens:
            vector = self._model.get_word_vector(token)
            if len(vector) != self.dimension:
                raise ValueError(
                    f"FastText returned dimension {len(vector)} for token {token!r}"
                )
            for index, value in enumerate(vector):
                summed[index] += float(value)

        averaged = [value / len(tokens) for value in summed]
        return _l2_normalize(averaged)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    @staticmethod
    def _resolve_model_path(
        model_path: str,
        repo_id: str,
        filename: str,
        auto_download: bool,
    ) -> Path:
        path = Path(model_path).expanduser()
        if path.is_file():
            return path
        if not auto_download:
            raise FileNotFoundError(
                f"FastText model not found at {path}. "
                "Download it or enable FASTTEXT_AUTO_DOWNLOAD."
            )
        return ensure_fasttext_model(
            destination=str(path),
            repo_id=repo_id,
            filename=filename,
        )


def _load_fasttext_model(path: str) -> Any:
    try:
        import fasttext
    except ImportError as exc:
        raise RuntimeError(
            "fasttext-wheel is required for FastText embeddings"
        ) from exc
    return fasttext.load_model(path)


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
