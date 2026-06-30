from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.shared.nlp.embeddings.training_dataset import (
    RetrievalTrainingExample,
    dataset_validation_report,
    examples_for_split,
    load_retrieval_training_dataset,
)


QUERY_PREFIX = "query: "
DOCUMENT_PREFIX = "passage: "


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    examples = load_retrieval_training_dataset(dataset_path)
    train_examples = examples_for_split(examples, "train")
    validation_examples = examples_for_split(examples, "validation")

    try:
        import torch
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
            losses,
        )
        from sentence_transformers.evaluation import InformationRetrievalEvaluator
        from sentence_transformers.training_args import BatchSamplers
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-training.txt before running fine-tuning"
        ) from exc

    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_sequence_length
    dimension = model.get_sentence_embedding_dimension()
    if dimension != 384:
        raise ValueError(
            f"Expected a 384-dimensional base encoder, got {dimension}. "
            "Changing the dimension also requires a different PGVector migration."
        )

    train_dataset = _to_huggingface_dataset(train_examples, Dataset)
    validation_dataset = _to_huggingface_dataset(validation_examples, Dataset)
    evaluator = _build_evaluator(
        validation_examples,
        InformationRetrievalEvaluator,
        name="validation",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir.parent / f"{output_dir.name}-checkpoints"
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=bool(torch.cuda.is_available()),
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=max(1, args.logging_steps),
        seed=args.seed,
        report_to="none",
        run_name="frimeet-multilingual-e5-finetuning",
    )
    loss = losses.MultipleNegativesRankingLoss(model)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    baseline_metrics = evaluator(model)
    trainer.train()
    finetuned_metrics = evaluator(model)
    model.save_pretrained(str(output_dir))

    metadata = {
        "base_model": args.base_model,
        "embedding_dimension": dimension,
        "query_prefix": QUERY_PREFIX,
        "document_prefix": DOCUMENT_PREFIX,
        "max_sequence_length": args.max_sequence_length,
        "dataset_sha256": _sha256(dataset_path),
        "dataset_report": asdict(dataset_validation_report(examples)),
        "baseline_metrics": baseline_metrics,
        "finetuned_metrics": finetuned_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
    }
    (output_dir / "frimeet_embedding_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=float))
    if args.hub_model_id:
        _upload_to_hub(output_dir, args.hub_model_id, args.private)


def _to_huggingface_dataset(
    examples: list[RetrievalTrainingExample],
    dataset_class,
):
    return dataset_class.from_dict(
        {
            "anchor": [f"{QUERY_PREFIX}{item.query}" for item in examples],
            "positive": [f"{DOCUMENT_PREFIX}{item.positive}" for item in examples],
            "negative": [f"{DOCUMENT_PREFIX}{item.negative}" for item in examples],
        }
    )


def _build_evaluator(examples, evaluator_class, name: str):
    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}
    for item in examples:
        queries[item.query_id] = f"{QUERY_PREFIX}{item.query}"
        corpus[item.positive_id] = f"{DOCUMENT_PREFIX}{item.positive}"
        corpus[item.negative_id] = f"{DOCUMENT_PREFIX}{item.negative}"
        relevant_docs.setdefault(item.query_id, set()).add(item.positive_id)
    return evaluator_class(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        accuracy_at_k=[1, 3, 5],
        precision_recall_at_k=[1, 3, 5],
        mrr_at_k=[10],
        ndcg_at_k=[5, 10],
        map_at_k=[10],
        show_progress_bar=True,
    )


def _upload_to_hub(output_dir: Path, model_id: str, private: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required to upload the model") from exc
    api = HfApi()
    api.create_repo(repo_id=model_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=model_id,
        repo_type="model",
        folder_path=str(output_dir),
        commit_message="Upload Frimeet retrieval encoder",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Frimeet's retrieval encoder.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--output-dir", default="artifacts/frimeet-e5-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hub-model-id")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
