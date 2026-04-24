# src/training/experiment_runner.py

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.training.phase_runner import run_phase


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_two_phase_experiment(
    *,
    exp_output_dir: str | Path,
    base_model_path: str | Path,
    train_dataset,
    eval_dataset,
    lora_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    phases: list[dict[str, Any]],
    sampler_seed: int,
    anchor_to_group: dict[int, list[int]] | None = None,
) -> dict[str, Any]:
    exp_output_dir = Path(exp_output_dir)
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    phase_results = []
    current_adapter_path: Path | None = None

    for phase in phases:
        phase_name = phase["phase_name"]
        sampler_mode = phase["sampler_mode"]
        phase_output_dir = exp_output_dir / phase_name

        phase_result = run_phase(
            phase_name=phase_name,
            sampler_mode=sampler_mode,
            epochs=phase["epochs"],
            learning_rate=phase["learning_rate"],
            phase_output_dir=phase_output_dir,
            base_model_path=base_model_path,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            lora_cfg=lora_cfg,
            training_cfg=training_cfg,
            sampler_seed=sampler_seed,
            anchor_to_group=anchor_to_group if sampler_mode == "grouped" else None,
            init_adapter_path=current_adapter_path,
        )
        phase_results.append(asdict(phase_result))

        current_adapter_path = phase_output_dir / "adapter"

    summary = {
        "n_phases": len(phase_results),
        "phase_results": phase_results,
        "final_adapter_path": str(current_adapter_path) if current_adapter_path else None,
    }

    save_json(summary, exp_output_dir / "run_summary.json")
    return summary