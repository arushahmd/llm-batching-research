# scripts/aggregate_results.py

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
REPORTS_ROOT = PROJECT_ROOT / "reports"

EXPERIMENT_BLOCKS = {
    "1k_short": range(12, 16),   # exp_012–015
    "1k_long": range(18, 22),    # exp_018–021
    "3k_long": range(22, 26),    # exp_022–025
    "5k_long": range(26, 30),    # exp_026–029
}


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def safe_std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return stdev(values)


def ci95_halfwidth(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return 1.96 * (stdev(values) / math.sqrt(len(values)))


def extract_seed_from_dirname(dirname: str) -> int | None:
    try:
        return int(dirname.split("_")[1])
    except Exception:
        return None


def extract_exp_num(exp_id: str) -> int | None:
    match = re.match(r"exp_(\d+)", exp_id)
    if not match:
        return None
    return int(match.group(1))


def infer_block(exp_id: str) -> str | None:
    exp_num = extract_exp_num(exp_id)
    if exp_num is None:
        return None

    for block_name, exp_range in EXPERIMENT_BLOCKS.items():
        if exp_num in exp_range:
            return block_name

    return None


def discover_experiment_ids() -> list[str]:
    if not EXPERIMENTS_ROOT.exists():
        return []

    exp_ids = []

    for exp_dir in sorted(EXPERIMENTS_ROOT.iterdir()):
        if not exp_dir.is_dir():
            continue
        if not exp_dir.name.startswith("exp_"):
            continue
        if infer_block(exp_dir.name) is None:
            continue
        if list(exp_dir.glob("seed_*/run_summary.json")):
            exp_ids.append(exp_dir.name)

    return exp_ids


def discover_seed_summaries(exp_id: str) -> list[Path]:
    return sorted((EXPERIMENTS_ROOT / exp_id).glob("seed_*/run_summary.json"))


def normalize_method_label(label: str) -> str:
    mapping = {
        "random": "Random",
        "grouped": "Grouped",
        "random→grouped": "Random→Grouped",
        "grouped→random": "Grouped→Random",
    }
    return mapping.get(label.lower(), label)


def infer_method_from_phase_results(phase_results: list[dict[str, Any]]) -> str:
    sampler_modes = [
        str(phase.get("sampler_mode", "")).strip().lower()
        for phase in phase_results
        if phase.get("sampler_mode")
    ]

    if not sampler_modes:
        return "Unknown"

    if len(sampler_modes) == 1:
        return normalize_method_label(sampler_modes[0])

    if sampler_modes[0] == sampler_modes[1]:
        return normalize_method_label(sampler_modes[0])

    return normalize_method_label(f"{sampler_modes[0]}→{sampler_modes[1]}")


def extract_record(exp_id: str, summary_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)

    block = infer_block(exp_id)
    seed = extract_seed_from_dirname(summary_path.parent.name)

    phase_results = summary.get("phase_results", [])
    method = infer_method_from_phase_results(phase_results)

    phase_1 = phase_results[0] if len(phase_results) > 0 else {}
    phase_2 = phase_results[1] if len(phase_results) > 1 else {}

    phase_1_train_metrics = phase_1.get("train_metrics", {})
    phase_1_eval_metrics = phase_1.get("eval_metrics", {})
    phase_2_train_metrics = phase_2.get("train_metrics", {})
    phase_2_eval_metrics = phase_2.get("eval_metrics", {})

    phase_1_train_loss = phase_1_train_metrics.get("train_loss")
    phase_1_eval_loss = phase_1_eval_metrics.get("eval_loss")
    phase_2_train_loss = phase_2_train_metrics.get("train_loss")
    phase_2_eval_loss = phase_2_eval_metrics.get("eval_loss")

    final_train_loss = phase_2_train_loss if phase_2_train_loss is not None else phase_1_train_loss
    final_eval_loss = phase_2_eval_loss if phase_2_eval_loss is not None else phase_1_eval_loss

    generalization_gap = None
    if final_eval_loss is not None and final_train_loss is not None:
        generalization_gap = float(final_eval_loss) - float(final_train_loss)

    phase_2_delta_eval = None
    if phase_1_eval_loss is not None and phase_2_eval_loss is not None:
        phase_2_delta_eval = float(phase_2_eval_loss) - float(phase_1_eval_loss)

    return {
        "block": block,
        "exp_id": exp_id,
        "method": method,
        "seed": seed,
        "phase_1_train_loss": phase_1_train_loss,
        "phase_1_eval_loss": phase_1_eval_loss,
        "phase_2_train_loss": phase_2_train_loss,
        "phase_2_eval_loss": phase_2_eval_loss,
        "final_eval_loss": final_eval_loss,
        "final_train_loss": final_train_loss,
        "generalization_gap": generalization_gap,
        "phase_2_delta_eval": phase_2_delta_eval,
        "summary_path": str(summary_path),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    eval_losses = [
        float(r["final_eval_loss"])
        for r in records
        if r.get("final_eval_loss") is not None
    ]
    train_losses = [
        float(r["final_train_loss"])
        for r in records
        if r.get("final_train_loss") is not None
    ]
    gaps = [
        float(r["generalization_gap"])
        for r in records
        if r.get("generalization_gap") is not None
    ]
    deltas = [
        float(r["phase_2_delta_eval"])
        for r in records
        if r.get("phase_2_delta_eval") is not None
    ]

    seeds = sorted({r["seed"] for r in records if r.get("seed") is not None})
    exp_ids = sorted({r["exp_id"] for r in records})

    return {
        "Block": records[0]["block"],
        "Method": records[0]["method"],
        "exp_ids": ",".join(exp_ids),
        "N": len(records),
        "Seeds": ",".join(str(s) for s in seeds),
        "Mean Eval Loss": safe_mean(eval_losses),
        "Std Eval Loss": safe_std(eval_losses),
        "95% CI Eval": ci95_halfwidth(eval_losses),
        "Mean Train Loss": safe_mean(train_losses),
        "Std Train Loss": safe_std(train_losses),
        "95% CI Train": ci95_halfwidth(train_losses),
        "Mean Gap": safe_mean(gaps),
        "Std Gap": safe_std(gaps),
        "Mean Phase2 ΔEval": safe_mean(deltas),
        "Std Phase2 ΔEval": safe_std(deltas),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    block_order = {
        "1k_short": 0,
        "1k_long": 1,
        "3k_long": 2,
        "5k_long": 3,
    }
    method_order = {
        "Random": 0,
        "Grouped": 1,
        "Grouped→Random": 2,
        "Random→Grouped": 3,
    }

    return sorted(
        rows,
        key=lambda r: (
            block_order.get(r["Block"], 999),
            method_order.get(r["Method"], 999),
        ),
    )


def main() -> None:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    experiment_ids = discover_experiment_ids()

    if not experiment_ids:
        raise SystemExit("No valid experiment folders found.")

    all_seed_rows: list[dict[str, Any]] = []
    grouped_records: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for exp_id in experiment_ids:
        summary_paths = discover_seed_summaries(exp_id)

        if not summary_paths:
            print(f"[WARN] No run_summary.json found for {exp_id}")
            continue

        for summary_path in summary_paths:
            record = extract_record(exp_id, summary_path)
            all_seed_rows.append(record)

            key = (record["block"], record["method"])
            grouped_records.setdefault(key, []).append(record)

    summary_rows = [
        aggregate_records(records)
        for records in grouped_records.values()
    ]
    summary_rows = sort_summary_rows(summary_rows)

    per_seed_csv = REPORTS_ROOT / "master_per_seed_results.csv"
    summary_csv = REPORTS_ROOT / "master_summary_table.csv"
    summary_json = REPORTS_ROOT / "master_summary.json"

    write_csv(
        per_seed_csv,
        all_seed_rows,
        fieldnames=[
            "block",
            "exp_id",
            "method",
            "seed",
            "phase_1_train_loss",
            "phase_1_eval_loss",
            "phase_2_train_loss",
            "phase_2_eval_loss",
            "final_eval_loss",
            "final_train_loss",
            "generalization_gap",
            "phase_2_delta_eval",
            "summary_path",
        ],
    )

    write_csv(
        summary_csv,
        summary_rows,
        fieldnames=[
            "Block",
            "Method",
            "exp_ids",
            "N",
            "Seeds",
            "Mean Eval Loss",
            "Std Eval Loss",
            "95% CI Eval",
            "Mean Train Loss",
            "Std Train Loss",
            "95% CI Train",
            "Mean Gap",
            "Std Gap",
            "Mean Phase2 ΔEval",
            "Std Phase2 ΔEval",
        ],
    )

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print("\nAggregation complete.")
    print(f"Detected experiments: {len(experiment_ids)}")
    print(f"Per-seed CSV: {per_seed_csv}")
    print(f"Summary CSV:  {summary_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()