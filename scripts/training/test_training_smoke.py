from __future__ import annotations

from pathlib import Path

from src.batching.grouping import build_anchor_to_group_map
from src.batching.index_loader import load_semantic_index
from src.data.alignment import build_alignment
from src.data.processed_loader import load_processed_dataset
from src.data.raw_loader import load_raw_dataset
from src.training.experiment_runner import run_two_phase_experiment


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    raw_dataset = load_raw_dataset(project_root / "data" / "raw" / "dolly_15k")
    processed_dataset = load_processed_dataset(
        project_root / "data" / "processed" / "dolly_small_1k__flan-t5-small__tok_with_rawidx_v1"
    )
    semantic_index = load_semantic_index(
        project_root / "data" / "indexes" / "semantic_groups" / "dolly_small_1k__all-MiniLM-L6-v2"
    )

    alignment = build_alignment(
        processed_train_ds=processed_dataset["train"],
        raw_train_ds=raw_dataset["train"],
    )

    small_train = processed_dataset["train"].select(range(64))
    small_eval = processed_dataset["eval"].select(range(16))

    anchor_to_group = build_anchor_to_group_map(
        alignment=alignment,
        semantic_index=semantic_index,
        max_group_size=8,
        include_anchor=True,
    )

    # restrict group map to selected training subset only
    valid_train_ids = set(range(len(small_train)))
    anchor_to_group_small = {
        k: [idx for idx in v if idx in valid_train_ids]
        for k, v in anchor_to_group.items()
        if k in valid_train_ids
    }

    lora_cfg = {
        "r": 16,
        "alpha": 32,
        "dropout": 0.05,
        "target_modules": ["q", "v"],
    }

    training_cfg = {
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "weight_decay": 0.01,
        "logging_steps": 5,
        "eval_strategy": "steps",
        "eval_steps": 5,
        "save_strategy": "no",
        "save_total_limit": 1,
        "predict_with_generate": False,
        "fp16": False,
        "bf16": False,
        "report_to": [],
    }

    phases = [
        {
            "phase_name": "phase_1_random",
            "sampler_mode": "random",
            "epochs": 0.2,
            "learning_rate": 5e-5,
        },
        {
            "phase_name": "phase_2_grouped",
            "sampler_mode": "grouped",
            "epochs": 0.2,
            "learning_rate": 5e-5,
        },
    ]

    summary = run_two_phase_experiment(
        exp_output_dir=project_root / "experiments" / "smoke_test_run",
        base_model_path=project_root / "models" / "flan-t5-small",
        train_dataset=small_train,
        eval_dataset=small_eval,
        lora_cfg=lora_cfg,
        training_cfg=training_cfg,
        phases=phases,
        sampler_seed=42,
        anchor_to_group=anchor_to_group_small,
    )

    print(summary)


if __name__ == "__main__":
    main()