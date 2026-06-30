from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.shared.nlp.embeddings.training_dataset import (
    dataset_validation_report,
    load_retrieval_training_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Frimeet retrieval JSONL data.")
    parser.add_argument("dataset")
    args = parser.parse_args()

    examples = load_retrieval_training_dataset(args.dataset)
    report = dataset_validation_report(examples)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
