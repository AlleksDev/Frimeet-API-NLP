"""Fine-tune a bi-encoder for open-vocabulary place retrieval.

Input is JSONL with at least ``query`` and ``positive``.  ``hard_negatives`` may
contain semantically close but incorrect place documents.  The script keeps all
runtime imports lazy so the API can still run with FastText during rollout.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)

    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError(
            "Install sentence-transformers and its PyTorch runtime before training"
        ) from exc

    rows = _read_training_rows(Path(args.train_file))
    examples = [
        InputExample(
            texts=[
                f"query: {row['query']}",
                f"passage: {row['positive']}",
                *(
                    f"passage: {negative}"
                    for negative in row["hard_negatives"][: args.max_hard_negatives]
                ),
            ]
        )
        for row in rows
    ]
    random.shuffle(examples)

    model = SentenceTransformer(args.base_model, device=args.device)
    data_loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=len(examples) >= args.batch_size,
    )
    cached_loss = getattr(losses, "CachedMultipleNegativesRankingLoss", None)
    if cached_loss is not None:
        train_loss = cached_loss(model, mini_batch_size=args.mini_batch_size)
    else:
        train_loss = losses.MultipleNegativesRankingLoss(model)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    warmup_steps = max(1, int(len(data_loader) * args.epochs * args.warmup_ratio))
    model.fit(
        train_objectives=[(data_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(output),
        show_progress_bar=True,
    )
    (output / "places_training_manifest.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "training_examples": len(examples),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "query_prefix": "query: ",
                "passage_prefix": "passage: ",
                "objective": type(train_loss).__name__,
                "seed": args.seed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _read_training_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Training dataset not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        payload = json.loads(raw_line)
        query = _required_text(payload.get("query"), line_number, "query")
        positive = _required_text(payload.get("positive"), line_number, "positive")
        raw_negatives = payload.get("hard_negatives", [])
        if isinstance(raw_negatives, str):
            raw_negatives = [raw_negatives]
        if not isinstance(raw_negatives, list):
            raise ValueError(
                f"Line {line_number}: hard_negatives must be a list of strings"
            )
        negatives = [
            _required_text(value, line_number, "hard_negatives")
            for value in raw_negatives
        ]
        rows.append(
            {"query": query, "positive": positive, "hard_negatives": negatives}
        )
    if len(rows) < 2:
        raise ValueError("At least two training examples are required")
    return rows


def _required_text(value: Any, line_number: int, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line_number}: {field} must be a non-empty string")
    return " ".join(value.split())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--base-model",
        default="intfloat/multilingual-e5-base",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mini-batch-size", type=int, default=8)
    parser.add_argument("--max-hard-negatives", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
