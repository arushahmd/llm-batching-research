# scripts/evaluation/evaluate_generation_quality.py

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import evaluate


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_train_dataset(raw_dataset_path: Path) -> Dataset:
    ds = load_from_disk(str(raw_dataset_path))
    if isinstance(ds, DatasetDict):
        return ds["train"]
    return ds


def reconstruct_eval_split(raw_train_ds: Dataset, train_ratio: float = 0.9, seed: int = 42) -> Dataset:
    split = raw_train_ds.train_test_split(test_size=1 - train_ratio, seed=seed)
    return split["test"]


def format_input(sample: dict[str, Any]) -> str:
    instruction = sample.get("instruction", "") or ""
    context = sample.get("context", "") or ""
    if context:
        return f"{instruction}\n\nContext: {context}"
    return instruction


def batch_generate(
    model,
    tokenizer,
    inputs: list[str],
    max_input_length: int = 512,
    max_new_tokens: int = 128,
    batch_size: int = 8,
) -> list[str]:
    all_outputs = []

    for i in range(0, len(inputs), batch_size):
        batch_texts = inputs[i:i + batch_size]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )

        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_outputs.extend(decoded)

    return all_outputs


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bert_scores = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
    )

    return {
        "rouge1": float(rouge_scores["rouge1"]),
        "rouge2": float(rouge_scores["rouge2"]),
        "rougeL": float(rouge_scores["rougeL"]),
        "rougeLsum": float(rouge_scores["rougeLsum"]),
        "bertscore_f1": float(sum(bert_scores["f1"]) / len(bert_scores["f1"])),
    }


def save_predictions_csv(
    out_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "idx",
        "instruction",
        "context",
        "reference",
        "prediction",
    ]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_single_run(
    exp_dir: Path,
    seed_dir: Path,
    raw_dataset_path: Path,
    base_model_path: Path,
    output_root: Path,
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    run_summary_path = seed_dir / "run_summary.json"
    run_summary = load_json(run_summary_path)

    final_adapter_path = Path(run_summary["final_adapter_path"])
    exp_name = exp_dir.name
    seed_name = seed_dir.name

    print(f"Evaluating {exp_name} / {seed_name}")
    print(f"Adapter: {final_adapter_path}")

    raw_train = load_raw_train_dataset(raw_dataset_path)
    eval_ds = reconstruct_eval_split(raw_train, train_ratio=0.9, seed=42)

    inputs = [format_input(x) for x in eval_ds]
    references = [x["response"] for x in eval_ds]

    print(f"Loading tokenizer/model from base: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
    base_model = AutoModelForSeq2SeqLM.from_pretrained(str(base_model_path))

    model = PeftModel.from_pretrained(base_model, str(final_adapter_path))
    model.eval()

    if torch.cuda.is_available():
        model = model.to("cuda")

    predictions = batch_generate(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    metrics = compute_metrics(predictions, references)

    out_dir = output_root / exp_name / seed_name
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    for idx, sample in enumerate(eval_ds):
        prediction_rows.append(
            {
                "idx": idx,
                "instruction": sample.get("instruction", ""),
                "context": sample.get("context", ""),
                "reference": sample.get("response", ""),
                "prediction": predictions[idx],
            }
        )

    save_predictions_csv(out_dir / "predictions.csv", prediction_rows)

    summary = {
        "exp_id": exp_name,
        "seed": seed_name,
        "final_adapter_path": str(final_adapter_path),
        "n_eval_samples": len(eval_ds),
        **metrics,
    }

    with open(out_dir / "generation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def save_aggregate(outputs: list[dict[str, Any]], output_root: Path) -> None:
    per_seed_csv = output_root / "generation_metrics_per_seed.csv"
    summary_json = output_root / "generation_metrics_summary.json"
    summary_csv = output_root / "generation_metrics_summary_table.csv"

    if not outputs:
        return

    fieldnames = list(outputs[0].keys())
    with open(per_seed_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outputs)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in outputs:
        grouped.setdefault(row["exp_id"], []).append(row)

    summary_rows = []
    metric_names = ["rouge1", "rouge2", "rougeL", "rougeLsum", "bertscore_f1"]

    for exp_id, rows in grouped.items():
        summary = {
            "exp_id": exp_id,
            "n_seeds": len(rows),
            "seeds": ",".join(r["seed"] for r in rows),
        }
        for metric in metric_names:
            values = [r[metric] for r in rows]
            mean_val = sum(values) / len(values)
            std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
            summary[f"mean_{metric}"] = mean_val
            summary[f"std_{metric}"] = std_val
        summary_rows.append(summary)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    fieldnames = list(summary_rows[0].keys())
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved: {per_seed_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_json}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-root", type=str, required=True)
    parser.add_argument("--raw-dataset-path", type=str, required=True)
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--exp-prefix", type=str, default="exp_026")
    parser.add_argument("--exp-ids", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root)
    raw_dataset_path = Path(args.raw_dataset_path)
    base_model_path = Path(args.base_model_path)
    output_root = Path(args.output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    all_exp_dirs = sorted([p for p in experiments_root.iterdir() if p.is_dir() and p.name.startswith("exp_")])

    if args.exp_ids:
        selected = set(args.exp_ids)
        exp_dirs = [p for p in all_exp_dirs if p.name in selected]
    else:
        exp_dirs = [p for p in all_exp_dirs if p.name.startswith(args.exp_prefix)]

    outputs = []

    for exp_dir in exp_dirs:
        seed_dirs = sorted([p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])
        for seed_dir in seed_dirs:
            if not (seed_dir / "run_summary.json").exists():
                continue
            result = evaluate_single_run(
                exp_dir=exp_dir,
                seed_dir=seed_dir,
                raw_dataset_path=raw_dataset_path,
                base_model_path=base_model_path,
                output_root=output_root,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            outputs.append(result)

    save_aggregate(outputs, output_root)


if __name__ == "__main__":
    main()