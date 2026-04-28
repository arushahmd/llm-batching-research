# scripts/training/run_experiment.py

from __future__ import annotations

import sys
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import datasets
import peft
import torch
import transformers

from src.batching.grouping import build_anchor_to_group_map
from src.batching.index_loader import load_semantic_index
from src.config.loader import load_experiment_and_manifest
from src.data.alignment import build_alignment, validate_alignment
from src.data.processed_loader import load_processed_dataset
from src.data.raw_loader import load_raw_dataset
from src.training.experiment_runner import run_two_phase_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_run_manifest(
    *,
    seed: int,
    exp_config: dict[str, Any],
    resolved_paths,
    run_output_dir: Path,
    raw_dataset,
    processed_dataset,
) -> dict[str, Any]:
    return {
        "created_at_utc": utc_now_iso(),
        "seed": seed,
        "exp_id": exp_config["experiment"]["exp_id"],
        "project_root": str(resolved_paths.project_root),
        "paths": {
            "base_model_path": str(resolved_paths.base_model_path),
            "embedding_model_path": str(resolved_paths.embedding_model_path),
            "raw_dataset_path": str(resolved_paths.raw_dataset_path),
            "processed_dataset_path": str(resolved_paths.processed_dataset_path),
            "semantic_index_path": str(resolved_paths.semantic_index_path),
            "run_output_dir": str(run_output_dir),
        },
        "dataset_info": {
            "raw_train_rows": len(raw_dataset["train"]),
            "processed_train_rows": len(processed_dataset["train"]),
            "processed_eval_rows": len(processed_dataset["eval"]),
        },
        "config_snapshot": exp_config,
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "datasets_version": datasets.__version__,
            "peft_version": peft.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one experiment config for one seed.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Single seed to run",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="manifests/project_manifest.json",
        help="Path to project manifest JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    exp_config, project_manifest, resolved_paths = load_experiment_and_manifest(
        experiment_config_path=args.config,
        project_manifest_path=args.manifest,
    )

    exp_id = exp_config["experiment"]["exp_id"]
    seed = args.seed

    run_output_dir = resolved_paths.experiments_root / exp_id / f"seed_{seed:03d}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    raw_dataset = load_raw_dataset(resolved_paths.raw_dataset_path)
    processed_dataset = load_processed_dataset(resolved_paths.processed_dataset_path)

    alignment = build_alignment(
        processed_train_ds=processed_dataset["train"],
        raw_train_ds=raw_dataset["train"],
    )
    alignment_summary = validate_alignment(alignment)

    phases = exp_config["phases"]
    has_grouped_phase = any(phase["sampler_mode"] == "grouped" for phase in phases)

    anchor_to_group = None
    semantic_index_summary = None

    if has_grouped_phase:
        semantic_index = load_semantic_index(resolved_paths.semantic_index_path)

        anchor_to_group = build_anchor_to_group_map(
            alignment=alignment,
            semantic_index=semantic_index,
            max_group_size=exp_config["grouping"]["max_group_size"],
            include_anchor=exp_config["grouping"].get("include_anchor", True),
        )

        semantic_index_summary = {
            "n_rows": semantic_index.n_rows,
            "top_k": semantic_index.top_k,
            "meta": semantic_index.meta,
        }

    run_manifest = build_run_manifest(
        seed=seed,
        exp_config=exp_config,
        resolved_paths=resolved_paths,
        run_output_dir=run_output_dir,
        raw_dataset=raw_dataset,
        processed_dataset=processed_dataset,
    )
    run_manifest["alignment_summary"] = alignment_summary
    if semantic_index_summary is not None:
        run_manifest["semantic_index_summary"] = semantic_index_summary

    save_json(run_manifest, run_output_dir / "run_manifest.json")

    summary = run_two_phase_experiment(
        exp_output_dir=run_output_dir,
        base_model_path=resolved_paths.base_model_path,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["eval"],
        lora_cfg=exp_config["lora"],
        training_cfg=exp_config["training"],
        phases=phases,
        sampler_seed=seed,
        anchor_to_group=anchor_to_group,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()