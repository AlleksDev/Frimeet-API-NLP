"""Fine-tune a Hugging Face token classifier for open place-chat slots.

The input files are JSONL.  Each non-empty line has this shape::

    {
      "text": "Quiero donas artesanales sin ruido",
      "spans": [
        {"start": 7, "end": 25, "slot": "CATEGORY"},
        {"start": 30, "end": 35, "slot": "EXCLUSION"}
      ]
    }

Offsets use Python's half-open character convention: ``text[start:end]``.
Only domain-independent slot types are accepted.  Category values remain raw
text, so this training path does not recreate a closed business taxonomy.

Imports for Transformers and PyTorch are deliberately lazy.  Importing this
module, validating data, and running ``--help`` do not require the optional ML
runtime.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SLOT_TYPES: tuple[str, ...] = (
    "CATEGORY",
    "PREFERENCE",
    "EXCLUSION",
    "LOCATION",
    "REFERENCE",
    "RADIUS",
)
LABELS: tuple[str, ...] = (
    "O",
    *(label for slot in SLOT_TYPES for label in (f"B-{slot}", f"I-{slot}")),
)
LABEL_TO_ID: Mapping[str, int] = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL: Mapping[int, str] = {index: label for label, index in LABEL_TO_ID.items()}
IGNORED_LABEL_ID = -100


@dataclass(frozen=True)
class LabeledSpan:
    """One validated, open-value slot annotation."""

    start: int
    end: int
    slot: str


@dataclass(frozen=True)
class IntentTrainingExample:
    """One validated token-classification example."""

    text: str
    spans: tuple[LabeledSpan, ...]


def validate_training_example(
    payload: Any,
    *,
    context: str = "example",
) -> IntentTrainingExample:
    """Validate one JSON-compatible example and normalize slot casing.

    Spans must be disjoint.  This is stricter than silently selecting one label
    for an overlapping token and makes annotation errors fail before training.
    """

    if not isinstance(payload, Mapping):
        raise ValueError(f"{context}: expected a JSON object")

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{context}: text must be a non-empty string")

    raw_spans = payload.get("spans")
    if not isinstance(raw_spans, list):
        raise ValueError(f"{context}: spans must be a list")

    spans: list[LabeledSpan] = []
    for index, raw_span in enumerate(raw_spans):
        span_context = f"{context}, span {index}"
        if not isinstance(raw_span, Mapping):
            raise ValueError(f"{span_context}: expected a JSON object")

        start = _integer_offset(raw_span.get("start"), "start", span_context)
        end = _integer_offset(raw_span.get("end"), "end", span_context)
        if start < 0 or end <= start or end > len(text):
            raise ValueError(
                f"{span_context}: offsets must satisfy "
                f"0 <= start < end <= {len(text)}; got start={start}, end={end}"
            )
        if not text[start:end].strip():
            raise ValueError(f"{span_context}: annotated text cannot be blank")

        raw_slot = raw_span.get("slot", raw_span.get("label"))
        if not isinstance(raw_slot, str) or not raw_slot.strip():
            raise ValueError(f"{span_context}: slot must be a non-empty string")
        slot = raw_slot.strip().upper()
        if slot not in SLOT_TYPES:
            raise ValueError(
                f"{span_context}: unsupported slot {raw_slot!r}; expected one of "
                + ", ".join(SLOT_TYPES)
            )
        spans.append(LabeledSpan(start=start, end=end, slot=slot))

    spans.sort(key=lambda span: (span.start, span.end, span.slot))
    for previous, current in zip(spans, spans[1:]):
        if current.start < previous.end:
            raise ValueError(
                f"{context}: overlapping spans "
                f"[{previous.start}, {previous.end}) and "
                f"[{current.start}, {current.end})"
            )

    return IntentTrainingExample(text=text, spans=tuple(spans))


def read_jsonl(path: Path) -> list[IntentTrainingExample]:
    """Read and validate a UTF-8 JSONL dataset with contextual errors."""

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    examples: list[IntentTrainingExample] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}, line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            examples.append(
                validate_training_example(
                    payload,
                    context=f"{path}, line {line_number}",
                )
            )

    if not examples:
        raise ValueError(f"Dataset has no examples: {path}")
    return examples


def align_spans_to_token_offsets(
    text: str,
    spans: Sequence[LabeledSpan],
    token_offsets: Sequence[Sequence[int]],
    *,
    label_to_id: Mapping[str, int] = LABEL_TO_ID,
) -> list[int]:
    """Align character spans to tokenizer offsets using IOB labels.

    Any token with a non-empty intersection with a span receives that slot.
    This handles subword tokenizers while preserving character-level source
    annotations.  Special/padding tokens, represented by ``(0, 0)``, receive
    the standard ``-100`` ignore label.

    Every annotated span must be covered by at least one token.  Consequently,
    truncation cannot silently turn a positive span into ``O``.
    """

    if not isinstance(text, str):
        raise TypeError("text must be str")

    _validate_label_mapping(label_to_id)
    normalized_spans = tuple(spans)
    _validate_span_objects(text, normalized_spans)

    seen_span_indexes: set[int] = set()
    labels: list[int] = []
    previous_token_start = -1
    for token_index, raw_offset in enumerate(token_offsets):
        token_start, token_end = _token_offset(raw_offset, token_index, len(text))
        if token_start == token_end == 0:
            labels.append(IGNORED_LABEL_ID)
            continue
        if token_start < previous_token_start:
            raise ValueError("token offsets must be ordered by start position")
        previous_token_start = token_start

        matching = [
            span_index
            for span_index, span in enumerate(normalized_spans)
            if token_start < span.end and span.start < token_end
        ]
        if len(matching) > 1:
            raise ValueError(
                f"token {token_index} [{token_start}, {token_end}) intersects "
                "multiple annotated spans"
            )
        if not matching:
            labels.append(label_to_id["O"])
            continue

        span_index = matching[0]
        span = normalized_spans[span_index]
        prefix = "I" if span_index in seen_span_indexes else "B"
        labels.append(label_to_id[f"{prefix}-{span.slot}"])
        seen_span_indexes.add(span_index)

    missing = [
        span
        for span_index, span in enumerate(normalized_spans)
        if span_index not in seen_span_indexes
    ]
    if missing:
        details = ", ".join(
            f"{span.slot}[{span.start}, {span.end})" for span in missing
        )
        raise ValueError(
            "annotated spans were not covered by tokenizer offsets "
            f"(possibly truncated): {details}"
        )
    return labels


def encode_examples(
    examples: Sequence[IntentTrainingExample],
    tokenizer: Any,
    *,
    max_length: int,
) -> list[dict[str, Any]]:
    """Tokenize and align examples without depending on a dataset library."""

    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")

    encoded_examples: list[dict[str, Any]] = []
    for example_index, example in enumerate(examples):
        encoding = tokenizer(
            example.text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        if not isinstance(encoding, Mapping):
            raise TypeError(
                f"tokenizer output for example {example_index} must be a mapping"
            )
        if "offset_mapping" not in encoding:
            raise ValueError(
                "tokenizer did not return offset_mapping; a fast tokenizer is required"
            )

        offsets = encoding["offset_mapping"]
        if not isinstance(offsets, Sequence) or isinstance(offsets, (str, bytes)):
            raise ValueError(
                f"tokenizer offset_mapping for example {example_index} "
                "must be a sequence"
            )
        feature = {
            key: value for key, value in encoding.items() if key != "offset_mapping"
        }
        feature["labels"] = align_spans_to_token_offsets(
            example.text,
            example.spans,
            offsets,
        )
        input_ids = feature.get("input_ids")
        if not isinstance(input_ids, Sequence) or isinstance(input_ids, (str, bytes)):
            raise ValueError(
                f"tokenizer output for example {example_index} has no input_ids sequence"
            )
        if len(input_ids) != len(feature["labels"]):
            raise ValueError(
                f"tokenizer output for example {example_index} has mismatched "
                "input_ids and offset_mapping lengths"
            )
        encoded_examples.append(feature)
    return encoded_examples


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _run_training(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, help="Training JSONL path")
    parser.add_argument(
        "--validation-file",
        help="Optional validation JSONL path",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--base-model",
        default="dccuchile/bert-base-spanish-wwm-cased",
        help="Hugging Face model id or local model directory",
    )
    parser.add_argument("--epochs", type=_positive_float, default=3.0)
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument("--learning-rate", type=_positive_float, default=2e-5)
    parser.add_argument("--max-length", type=_positive_int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _run_training(args: argparse.Namespace) -> None:
    # Optional heavyweight imports stay behind argument parsing so ``--help``
    # remains available in API and CI environments without the ML toolchain.
    try:
        import transformers
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training requires transformers with its PyTorch backend. "
            "Install the optional ML dependencies before running this script."
        ) from exc

    train_path = Path(args.train_file)
    validation_path = Path(args.validation_file) if args.validation_file else None
    train_examples = read_jsonl(train_path)
    validation_examples = read_jsonl(validation_path) if validation_path else []

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "The selected model does not provide a fast tokenizer with offsets"
        )

    encoded_train = encode_examples(
        train_examples,
        tokenizer,
        max_length=args.max_length,
    )
    encoded_validation = encode_examples(
        validation_examples,
        tokenizer,
        max_length=args.max_length,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        label2id=dict(LABEL_TO_ID),
        id2label=dict(ID_TO_LABEL),
        ignore_mismatched_sizes=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    has_validation = bool(encoded_validation)
    strategy = "epoch" if has_validation else "no"
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "data_seed": args.seed,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 25,
        "report_to": [],
        "load_best_model_at_end": has_validation,
    }
    parameter_names = inspect.signature(TrainingArguments.__init__).parameters
    strategy_parameter = (
        "eval_strategy" if "eval_strategy" in parameter_names else "evaluation_strategy"
    )
    training_kwargs[strategy_parameter] = strategy

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=encoded_train,
        eval_dataset=encoded_validation if has_validation else None,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        tokenizer=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    manifest = {
        "schema_version": 1,
        "task": "token-classification",
        "architecture": "bert-open-place-intent",
        "base_model": args.base_model,
        "slots": list(SLOT_TYPES),
        "labels": list(LABELS),
        "label_to_id": dict(LABEL_TO_ID),
        "training_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
        "train_loss": _finite_or_none(getattr(train_result, "training_loss", None)),
        "transformers_version": getattr(transformers, "__version__", "unknown"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "place_intent_training_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _integer_offset(value: Any, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}: {field} must be an integer")
    return value


def _validate_span_objects(text: str, spans: Sequence[LabeledSpan]) -> None:
    previous: LabeledSpan | None = None
    for index, span in enumerate(spans):
        if not isinstance(span, LabeledSpan):
            raise TypeError(f"span {index} must be LabeledSpan")
        if span.slot not in SLOT_TYPES:
            raise ValueError(f"span {index} has unsupported slot {span.slot!r}")
        if span.start < 0 or span.end <= span.start or span.end > len(text):
            raise ValueError(f"span {index} has invalid offsets")
        if previous is not None and span.start < previous.end:
            raise ValueError("spans must be sorted and non-overlapping")
        previous = span


def _validate_label_mapping(label_to_id: Mapping[str, int]) -> None:
    missing = [label for label in LABELS if label not in label_to_id]
    if missing:
        raise ValueError("label_to_id is missing labels: " + ", ".join(missing))


def _token_offset(
    raw_offset: Sequence[int],
    token_index: int,
    text_length: int,
) -> tuple[int, int]:
    if (
        isinstance(raw_offset, (str, bytes))
        or not isinstance(raw_offset, Sequence)
        or len(raw_offset) != 2
    ):
        raise ValueError(f"token offset {token_index} must contain start and end")
    start, end = raw_offset
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise ValueError(f"token offset {token_index} must contain integers")
    if start == end == 0:
        return 0, 0
    if start < 0 or end <= start or end > text_length:
        raise ValueError(
            f"token offset {token_index} is invalid for text length {text_length}"
        )
    return start, end


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite value greater than zero")
    return parsed


def _finite_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


if __name__ == "__main__":
    main()
