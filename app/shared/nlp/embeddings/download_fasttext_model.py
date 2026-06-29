import argparse
from pathlib import Path
import shutil


DEFAULT_REPO_ID = "facebook/fasttext-es-vectors"
DEFAULT_FILENAME = "model.bin"
DEFAULT_DESTINATION = "/opt/models/fasttext-es/model.bin"


def ensure_fasttext_model(
    destination: str,
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_FILENAME,
) -> Path:
    target = Path(destination).expanduser()
    if target.is_file():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required to download the FastText model"
        ) from exc

    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target.parent,
        )
    )
    if downloaded.resolve() == target.resolve():
        return target
    temporary_target = target.with_suffix(target.suffix + ".part")
    shutil.copyfile(downloaded, temporary_target)
    temporary_target.replace(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Spanish FastText model.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--destination", default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    path = ensure_fasttext_model(
        destination=args.destination,
        repo_id=args.repo_id,
        filename=args.filename,
    )
    print(path)


if __name__ == "__main__":
    main()
