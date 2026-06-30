from __future__ import annotations

import argparse
import csv
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


APPROVED = {"approved", "approve", "aprobado", "aprobar", "si", "sí", "yes"}
REJECTED = {"rejected", "reject", "rechazado", "rechazar", "no"}


def main() -> None:
    args = _parse_args()
    labels = _read_jsonl(Path(args.labels))
    decisions = _read_decisions(Path(args.review_csv))

    approved_rows: list[dict[str, str]] = []
    pending: list[tuple[str, str, str]] = []
    for row in labels:
        key = _row_key(row)
        status = decisions.get(key, "pending")
        if status in APPROVED:
            approved_rows.append(row)
        elif status in REJECTED:
            continue
        else:
            pending.append(key)

    if pending:
        preview = ", ".join("/".join(item) for item in pending[:5])
        raise ValueError(
            f"There are {len(pending)} pending or invalid review decisions. "
            f"First rows: {preview}"
        )
    if not approved_rows:
        raise ValueError("No approved rows remain after review")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        for row in approved_rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")

    examples = load_retrieval_training_dataset(output)
    report = dataset_validation_report(examples)
    print(
        json.dumps(
            {
                "output": str(output),
                "approved_rows": report.total_rows,
                "rows_by_split": report.rows_by_split,
                "unique_queries_by_split": report.unique_queries_by_split,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_decisions(path: Path) -> dict[tuple[str, str, str], str]:
    decisions: dict[tuple[str, str, str], str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            key = _row_key(row)
            if key in decisions:
                raise ValueError(f"Duplicate review row: {'/'.join(key)}")
            decisions[key] = str(row.get("review_status") or "pending").strip().casefold()
    return decisions


def _row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("query_id") or "").strip(),
        str(row.get("positive_id") or "").strip(),
        str(row.get("negative_id") or "").strip(),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a final retrieval dataset from reviewed weak labels."
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
