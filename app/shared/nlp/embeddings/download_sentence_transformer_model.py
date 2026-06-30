from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re


def sentence_transformer_cache_path(
    cache_dir: str,
    repo_id: str,
    revision: str,
) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "--", repo_id).strip("-")
    fingerprint = hashlib.sha256(f"{repo_id}@{revision}".encode("utf-8")).hexdigest()[:12]
    return Path(cache_dir).expanduser() / f"{safe_name}-{fingerprint}"


def ensure_sentence_transformer_model(
    cache_dir: str,
    repo_id: str,
    revision: str = "main",
    auto_download: bool = True,
) -> Path:
    destination = sentence_transformer_cache_path(cache_dir, repo_id, revision)
    if _is_complete_model(destination):
        return destination
    if not auto_download:
        raise FileNotFoundError(
            f"Sentence Transformer model {repo_id}@{revision} was not found at "
            f"{destination}. Preload it or enable SENTENCE_TRANSFORMER_AUTO_DOWNLOAD."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required to download the encoder") from exc

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(destination),
        token=os.getenv("HF_TOKEN") or None,
        ignore_patterns=[
            "onnx/*",
            "openvino/*",
            "pytorch_model.bin",
            "*.onnx",
        ],
    )
    if not _is_complete_model(destination):
        raise RuntimeError(
            f"The downloaded repository {repo_id}@{revision} is not a valid "
            "Sentence Transformers model (modules.json is missing)."
        )
    return destination


def _is_complete_model(path: Path) -> bool:
    return path.is_dir() and (path / "modules.json").is_file()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preload a Sentence Transformers model.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = ensure_sentence_transformer_model(
        cache_dir=args.cache_dir,
        repo_id=args.repo_id,
        revision=args.revision,
    )
    print(path)


if __name__ == "__main__":
    main()
