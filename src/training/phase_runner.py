# src/training/phase_runner.py

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.training.trainer_factory import build_trainer


@dataclass(frozen=True)
class PhaseRunResult:
    phase_name: str
    sampler_mode: str
    epochs: float
    learning_rate: float
    train_metrics: dict[str, Any]
    eval_metrics: dict[str, Any]
    output_dir: str


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def run_phase(
    *,
    phase_name: str,
    sampler_mode: str,
    epochs: float,
    learning_rate: float,
    phase_output_dir: str | Path,
    base_model_path: str | Path,
    train_dataset,
    eval_dataset,
    lora_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    sampler_seed: int,
    anchor_to_group: dict[int, list[int]] | None = None,
    init_adapter_path: str | Path | None = None,
) -> PhaseRunResult:
    components = build_trainer(
        base_model_path=base_model_path,
        output_dir=phase_output_dir,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        lora_cfg=lora_cfg,
        training_cfg=training_cfg,
        sampler_mode=sampler_mode,
        sampler_seed=sampler_seed,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        anchor_to_group=anchor_to_group,
        init_adapter_path=init_adapter_path,
    )

    trainer = components.trainer

    train_metrics = trainer.train().metrics
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(Path(phase_output_dir) / "adapter"))

    result = PhaseRunResult(
        phase_name=phase_name,
        sampler_mode=sampler_mode,
        epochs=epochs,
        learning_rate=learning_rate,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        output_dir=str(phase_output_dir),
    )

    save_json(asdict(result), Path(phase_output_dir) / "phase_summary.json")
    return result