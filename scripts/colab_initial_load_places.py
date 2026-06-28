"""Carga inicial de embeddings FastText de lugares desde Google Colab.

Ejecutar desde la raiz de un clon de este repositorio:

    !python scripts/colab_initial_load_places.py

El script lee credenciales desde los Secrets de Colab usando los mismos nombres
de las variables de entorno. Nunca imprime los valores secretos.
"""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
import socket
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REPOSITORY_URL = "https://github.com/AlleksDev/Frimeet-API-NLP.git"
REPOSITORY_BRANCH = "hf-deploy"
COLAB_REPOSITORY_PATH = Path("/content/Frimeet-API-NLP")


def _find_repo_root() -> Path:
    candidates: list[Path] = []
    if "__file__" in globals():
        candidates.append(Path(__file__).resolve().parents[1])

    current_directory = Path.cwd().resolve()
    candidates.extend(
        [
            current_directory,
            current_directory / "Frimeet-API-NLP",
        ]
    )
    candidates.extend(current_directory.parents)

    for candidate in candidates:
        if _is_repository_root(candidate):
            return candidate

    if COLAB_REPOSITORY_PATH.exists():
        raise RuntimeError(
            f"Existe {COLAB_REPOSITORY_PATH}, pero no contiene un clon valido. "
            "Reinicia el runtime de Colab o elimina esa carpeta incompleta."
        )

    print("No se encontro el repositorio; clonando la rama hf-deploy...")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            REPOSITORY_BRANCH,
            REPOSITORY_URL,
            str(COLAB_REPOSITORY_PATH),
        ],
        check=True,
    )
    if not _is_repository_root(COLAB_REPOSITORY_PATH):
        raise RuntimeError("El repositorio se clono, pero su estructura no es valida.")
    return COLAB_REPOSITORY_PATH


def _is_repository_root(path: Path) -> bool:
    return (path / "requirements.txt").is_file() and (path / "app").is_dir()


REPO_ROOT = _find_repo_root()
DEFAULT_MODEL_PATH = "/content/fasttext-es/model.bin"


def main() -> None:
    args = _parse_args()
    os.chdir(REPO_ROOT)
    _configure_environment()

    if not args.skip_install:
        print("[1/4] Instalando dependencias del proyecto...", flush=True)
        _run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "-r",
            str(REPO_ROOT / "requirements.txt"),
        )
    else:
        print("[1/4] Reutilizando dependencias instaladas.", flush=True)

    print("[2/4] Verificando API principal y acceso de red a RDS...", flush=True)
    _check_main_api()
    _check_pgvector_network()

    if not args.skip_download:
        print("[3/4] Descargando o reutilizando el modelo FastText...", flush=True)
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
    else:
        print("[3/4] Reutilizando el modelo FastText descargado.", flush=True)

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

    print(
        "[4/4] Cargando FastText y sincronizando lugares. "
        "La carga inicial del modelo puede tardar varios minutos...",
        flush=True,
    )
    _run(*command)
    print("Carga de lugares terminada correctamente.")


def _configure_environment() -> None:
    defaults = {
        "ENV": "colab",
        "MAIN_API_BASE_URL": "http://3.212.166.108",
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
        # A raw notebook cell shares os.environ with every previous execution.
        # Use a Colab Secret when explicitly configured; otherwise reset the
        # value to this script's current default instead of inheriting stale data.
        os.environ[name] = _read_colab_secret(name) or default

    os.environ["PGVECTOR_WRITER_PASSWORD"] = _read_required_secret(
        "PGVECTOR_WRITER_PASSWORD"
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
) -> str:
    value = os.getenv(name) or _read_colab_secret(name) or default
    return value or ""


def _read_required_secret(name: str) -> str:
    value = _read_setting(name)
    if value:
        return value
    value = getpass.getpass(f"Escribe {name} (la entrada permanecera oculta): ").strip()
    if not value:
        raise RuntimeError(f"No se proporciono el valor requerido {name!r}.")
    return value


def _read_colab_secret(name: str) -> str | None:
    try:
        from google.colab import userdata

        value = userdata.get(name)
        return str(value).strip() if value else None
    except Exception:
        return None


def _check_main_api() -> None:
    base_url = os.environ["MAIN_API_BASE_URL"].rstrip("/")
    path = os.environ["MAIN_API_PLACES_SEARCH_PATH"]
    url = f"{base_url}/{path.lstrip('/')}?{urlencode({'limit': 1})}"
    headers: dict[str, str] = {}
    token = os.getenv("MAIN_API_INTERNAL_TOKEN") or os.getenv("MAIN_API_AUTH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        with urlopen(Request(url, headers=headers), timeout=30) as response:
            status = response.status
    except HTTPError as exc:
        raise RuntimeError(
            f"La API principal respondio HTTP {exc.code} en {url}. "
            "Revisa MAIN_API_INTERNAL_TOKEN y MAIN_API_BASE_URL."
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Colab no pudo conectarse a la API principal {url}: {exc.reason}"
        ) from exc

    if status >= 400:
        raise RuntimeError(f"La API principal respondio HTTP {status} en {url}.")
    print(f"      API principal accesible (HTTP {status}).", flush=True)


def _check_pgvector_network() -> None:
    host = os.environ["PGVECTOR_HOST"]
    port = int(os.environ["PGVECTOR_PORT"])
    try:
        with socket.create_connection((host, port), timeout=15):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Colab no puede abrir una conexion TCP a {host}:{port}. "
            "Agrega temporalmente la IP publica de este runtime como /32 en el "
            "Security Group de RDS y confirma que la instancia sea accesible."
        ) from exc
    print(f"      RDS accesible por red en {host}:{port}.", flush=True)


def _run(*command: str) -> None:
    try:
        subprocess.run(list(command), cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        executable = " ".join(command)
        raise RuntimeError(
            f"Fallo el comando con codigo {exc.returncode}: {executable}. "
            "Revisa la salida inmediatamente anterior; si API y red aparecen OK, "
            "verifica la password/permisos de nlp_writer y la migracion VECTOR(300)."
        ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--page-limit", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    arguments = None if "__file__" in globals() else []
    return parser.parse_args(arguments)


if __name__ == "__main__":
    main()
