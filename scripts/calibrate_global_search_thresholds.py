"""Calibrate per-resource global-search thresholds from labeled score samples.

Input is JSONL with: resource_type, relevant, semantic_score and lexical_score.
At least one score must be present in every row. The script never reads source text or
private metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RESOURCE_TYPES = ("places", "posts", "users", "clubs", "groups", "events")


@dataclass(frozen=True)
class LabeledScore:
    resource_type: str
    relevant: bool
    semantic_score: float | None
    lexical_score: float | None


def calibrate_resource(
    samples: list[LabeledScore],
    target_precision: float = 0.95,
    default_semantic: float = 0.30,
    default_lexical: float = 0.05,
) -> dict[str, float | int]:
    positives = sum(sample.relevant for sample in samples)
    negatives = len(samples) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("each resource requires positive and negative labeled samples")

    semantic_candidates = _candidate_values(
        (sample.semantic_score for sample in samples),
        default=default_semantic,
        minimum=-1.0,
        maximum=1.0,
    )
    lexical_candidates = _candidate_values(
        (sample.lexical_score for sample in samples),
        default=default_lexical,
        minimum=0.0,
        maximum=None,
    )

    evaluated: list[dict[str, float | int]] = []
    for semantic_min in semantic_candidates:
        for lexical_min in lexical_candidates:
            tp = fp = fn = 0
            for sample in samples:
                accepted = (
                    sample.semantic_score is not None
                    and sample.semantic_score >= semantic_min
                ) or (
                    sample.lexical_score is not None
                    and sample.lexical_score >= lexical_min
                )
                if accepted and sample.relevant:
                    tp += 1
                elif accepted:
                    fp += 1
                elif sample.relevant:
                    fn += 1
            precision = tp / (tp + fp) if tp + fp else 1.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            evaluated.append(
                {
                    "semantic_min": semantic_min,
                    "lexical_min": lexical_min,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                }
            )

    precision_qualified = [
        result for result in evaluated if result["precision"] >= target_precision
    ]
    pool = precision_qualified or evaluated
    return max(
        pool,
        key=lambda result: (
            result["recall"] if precision_qualified else result["f1"],
            result["precision"],
            result["f1"],
            -result["semantic_min"],
            -result["lexical_min"],
        ),
    )


def load_samples(path: Path) -> list[LabeledScore]:
    samples: list[LabeledScore] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
            resource_type = str(payload["resource_type"])
            relevant = payload["relevant"]
            if resource_type not in RESOURCE_TYPES or not isinstance(relevant, bool):
                raise ValueError("invalid resource_type or relevant label")
            semantic = _optional_finite_float(payload.get("semantic_score"))
            lexical = _optional_finite_float(payload.get("lexical_score"))
            if semantic is None and lexical is None:
                raise ValueError("at least one score is required")
            if semantic is not None and not -1.0 <= semantic <= 1.0:
                raise ValueError("semantic_score must be between -1 and 1")
            if lexical is not None and lexical < 0.0:
                raise ValueError("lexical_score must be non-negative")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid calibration row {line_number}: {exc}") from exc
        samples.append(LabeledScore(resource_type, relevant, semantic, lexical))
    return samples


def _optional_finite_float(value: object) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("scores must be finite")
    return parsed


def _candidate_values(
    values: Iterable[float | None],
    *,
    default: float,
    minimum: float,
    maximum: float | None,
) -> list[float]:
    observed = sorted({value for value in values if value is not None})
    if len(observed) > 200:
        observed = [
            observed[round(index * (len(observed) - 1) / 199)]
            for index in range(200)
        ]
    candidates = {default, minimum, *observed}
    if observed:
        above_max = math.nextafter(observed[-1], math.inf)
        if maximum is None or above_max <= maximum:
            candidates.add(above_max)
    if maximum is not None:
        candidates.add(maximum)
    return sorted(
        value
        for value in candidates
        if value >= minimum and (maximum is None or value <= maximum)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 < args.target_precision <= 1.0:
        raise SystemExit("--target-precision must be in (0, 1]")

    samples = load_samples(args.input)
    grouped = {
        resource_type: [
            sample for sample in samples if sample.resource_type == resource_type
        ]
        for resource_type in RESOURCE_TYPES
    }
    missing = [resource for resource, items in grouped.items() if not items]
    if missing and not args.allow_partial:
        raise SystemExit(f"missing labeled resources: {', '.join(missing)}")

    results = {
        resource: calibrate_resource(items, target_precision=args.target_precision)
        for resource, items in grouped.items()
        if items
    }
    output = {
        "policy_version": args.policy_version,
        "target_precision": args.target_precision,
        "resource_thresholds": {
            resource: {
                "semantic_min": result["semantic_min"],
                "lexical_min": result["lexical_min"],
            }
            for resource, result in results.items()
        },
        "metrics": results,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
