"""Carga inicial de embeddings FastText de lugares desde Google Colab.

Ejecutar desde la raiz de un clon de este repositorio:

    !python scripts/colab_initial_load_places.py

El script lee credenciales desde los Secrets de Colab usando los mismos nombres
de las variables de entorno. Nunca imprime los valores secretos.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "/content/fasttext-es/model.bin"


def main() -> None:
    args = _parse_args()
    os.chdir(REPO_ROOT)
    _configure_environment()

    if not args.skip_install:
        _run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "-r",
            str(REPO_ROOT / "requirements.txt"),
        )

    if not args.skip_download:
        _run(
            sys.executable,
            "-m",
            "app.shared.nlp.embeddings.download_fasttext_model",
            "--repo-id",
            os.environ["FASTTEXT_MODEL_REPO_ID"],
            "--filename",
            os.environ["FASTTEXT_MODEL_FILENAME"],
            "--destination",
            os.environ["FASTTEXT_MODEL_PATH"],
        )

    command = [
        sys.executable,
        "-m",
        "app.jobs.initial_load_place_embeddings",
        "--batch-size",
        str(args.batch_size),
        "--page-limit",
        str(args.page_limit),
    ]
    if args.max_pages is not None:
        command.extend(["--max-pages", str(args.max_pages)])
    if args.dry_run:
        command.append("--dry-run")

    _run(*command)
    print("Carga de lugares terminada correctamente.")


def _configure_environment() -> None:
    defaults = {
        "ENV": "colab",
        "MAIN_API_BASE_URL": "http://52.86.8.11",
        "MAIN_API_PLACES_SEARCH_PATH": "/api/v1/places/search",
        "MAIN_API_TIMEOUT_SECONDS": "60",
        "MAIN_API_PLACES_PAGE_LIMIT": "50",
        "MAIN_API_PLACES_PAGINATION_MODE": "cursor",
        "VECTOR_STORE_PROVIDER": "aws_pgvector",
        "PGVECTOR_HOST": "nlp-vector-db.c2jwncm87zsa.us-east-1.rds.amazonaws.com",
        "PGVECTOR_PORT": "5432",
        "PGVECTOR_DATABASE": "nlp_vectors",
        "PGVECTOR_WRITER_USER": "nlp_writer",
        "PGVECTOR_SSL_MODE": "require",
        "EMBEDDING_PROVIDER": "fasttext",
        "EMBEDDING_DIMENSION": "300",
        "EMBEDDING_MODEL": "facebook/fasttext-es-vectors",
        "EMBEDDING_VERSION": "common-crawl-300-v1",
        "FASTTEXT_MODEL_PATH": DEFAULT_MODEL_PATH,
        "FASTTEXT_MODEL_REPO_ID": "facebook/fasttext-es-vectors",
        "FASTTEXT_MODEL_FILENAME": "model.bin",
        "FASTTEXT_AUTO_DOWNLOAD": "false",
        "LOG_LEVEL": "INFO",
    }
    for name, default in defaults.items():
        os.environ[name] = _read_setting(name, default=default)

    os.environ["PGVECTOR_WRITER_PASSWORD"] = _read_setting(
        "PGVECTOR_WRITER_PASSWORD",
        required=True,
    )

    for optional_name in (
        "MAIN_API_INTERNAL_TOKEN",
        "MAIN_API_AUTH_TOKEN",
        "HF_TOKEN",
    ):
        value = _read_setting(optional_name)
        if value:
            os.environ[optional_name] = value


def _read_setting(
    name: str,
    default: str | None = None,
    required: bool = False,
) -> str:
    value = os.getenv(name) or _read_colab_secret(name) or default
    if required and not value:
        raise RuntimeError(
            f"Falta el Secret {name!r} en Colab. Activa tambien Notebook access."
        )
    return value or ""


def _read_colab_secret(name: str) -> str | None:
    try:
        from google.colab import userdata

        value = userdata.get(name)
        return str(value).strip() if value else None
    except Exception:
        return None


def _run(*command: str) -> None:
    subprocess.run(list(command), cwd=REPO_ROOT, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--page-limit", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
