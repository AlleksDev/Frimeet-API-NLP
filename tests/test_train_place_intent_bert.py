import json

import pytest

from scripts.train_place_intent_bert import (
    IGNORED_LABEL_ID,
    LABEL_TO_ID,
    IntentTrainingExample,
    LabeledSpan,
    align_spans_to_token_offsets,
    build_parser,
    encode_examples,
    read_jsonl,
    validate_training_example,
)


def test_validate_training_example_preserves_raw_values_and_normalizes_slots() -> None:
    text = "Quiero donas artesanales sin ruido cerca de Parque Mexico"
    payload = {
        "text": text,
        "spans": [
            {
                "start": text.index("donas"),
                "end": text.index("artesanales") + len("artesanales"),
                "slot": "category",
            },
            {
                "start": text.index("ruido"),
                "end": text.index("ruido") + len("ruido"),
                "label": "EXCLUSION",
            },
            {
                "start": text.index("Parque"),
                "end": len(text),
                "slot": "LOCATION",
            },
        ],
    }

    example = validate_training_example(payload, context="line 1")

    assert example.text == text
    assert [span.slot for span in example.spans] == [
        "CATEGORY",
        "EXCLUSION",
        "LOCATION",
    ]
    assert [text[span.start : span.end] for span in example.spans] == [
        "donas artesanales",
        "ruido",
        "Parque Mexico",
    ]


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"text": "donas", "spans": "CATEGORY"}, "spans must be a list"),
        (
            {
                "text": "donas",
                "spans": [{"start": -1, "end": 5, "slot": "CATEGORY"}],
            },
            "offsets must satisfy",
        ),
        (
            {
                "text": "donas",
                "spans": [{"start": 0, "end": 5, "slot": "RESTAURANT"}],
            },
            "unsupported slot",
        ),
        (
            {
                "text": "donas ricas",
                "spans": [
                    {"start": 0, "end": 7, "slot": "CATEGORY"},
                    {"start": 6, "end": 11, "slot": "PREFERENCE"},
                ],
            },
            "overlapping spans",
        ),
    ],
)
def test_validate_training_example_rejects_invalid_annotations(
    payload: object,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        validate_training_example(payload)


def test_alignment_assigns_iob_to_subwords_and_ignores_special_tokens() -> None:
    text = "Quiero donas sin ruido"
    spans = (
        LabeledSpan(start=7, end=12, slot="CATEGORY"),
        LabeledSpan(start=17, end=22, slot="EXCLUSION"),
    )
    offsets = (
        (0, 0),
        (0, 6),
        (7, 9),
        (9, 12),
        (13, 16),
        (17, 22),
        (0, 0),
    )

    labels = align_spans_to_token_offsets(text, spans, offsets)

    assert labels == [
        IGNORED_LABEL_ID,
        LABEL_TO_ID["O"],
        LABEL_TO_ID["B-CATEGORY"],
        LABEL_TO_ID["I-CATEGORY"],
        LABEL_TO_ID["O"],
        LABEL_TO_ID["B-EXCLUSION"],
        IGNORED_LABEL_ID,
    ]


def test_alignment_rejects_annotations_lost_by_truncation() -> None:
    text = "donas cerca del parque"
    spans = (LabeledSpan(start=16, end=22, slot="LOCATION"),)

    with pytest.raises(ValueError, match="possibly truncated.*LOCATION"):
        align_spans_to_token_offsets(
            text,
            spans,
            [(0, 0), (0, 5), (6, 11), (0, 0)],
        )


class FakeFastTokenizer:
    is_fast = True

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        self.calls.append({"text": text, **kwargs})
        return {
            "input_ids": [101, 1001, 102],
            "attention_mask": [1, 1, 1],
            "offset_mapping": [(0, 0), (0, len(text)), (0, 0)],
        }


def test_encode_examples_works_with_an_injected_tokenizer() -> None:
    tokenizer = FakeFastTokenizer()
    examples = [
        IntentTrainingExample(
            text="cafecito",
            spans=(LabeledSpan(start=0, end=8, slot="CATEGORY"),),
        )
    ]

    encoded = encode_examples(examples, tokenizer, max_length=32)

    assert encoded == [
        {
            "input_ids": [101, 1001, 102],
            "attention_mask": [1, 1, 1],
            "labels": [
                IGNORED_LABEL_ID,
                LABEL_TO_ID["B-CATEGORY"],
                IGNORED_LABEL_ID,
            ],
        }
    ]
    assert tokenizer.calls == [
        {
            "text": "cafecito",
            "truncation": True,
            "max_length": 32,
            "return_offsets_mapping": True,
        }
    ]


def test_read_jsonl_reports_the_failing_line(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        json.dumps({"text": "donas", "spans": []})
        + "\n"
        + "{invalid-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"train\.jsonl, line 2: invalid JSON"):
        read_jsonl(dataset)


def test_help_is_built_without_importing_optional_ml_dependencies() -> None:
    help_text = build_parser().format_help()

    assert "--train-file" in help_text
    assert "--validation-file" in help_text
    assert "--base-model" in help_text
    assert "--learning-rate" in help_text
