# scripts/training/run_multiseed.py

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level dict in YAML: {path}")

    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multiseed experiment.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="manifests/project_manifest.json",
        help="Path to project manifest JSON",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run seeds even if run_summary.json already exists",
    )
    return parser.parse_args()


def get_exp_id(config: dict) -> str:
    try:
        return config["experiment"]["exp_id"]
    except KeyError as exc:
        raise ValueError("Experiment config missing experiment.exp_id") from exc


def get_seed_list(config: dict) -> list[int]:
    try:
        seeds = config["multiseed"]["seeds"]
    except KeyError as exc:
        raise ValueError("Experiment config missing multiseed.seeds") from exc

    if not isinstance(seeds, list) or not seeds:
        raise ValueError("multiseed.seeds must be a non-empty list")

    return [int(seed) for seed in seeds]


def seed_run_summary_path(exp_id: str, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "experiments"
        / exp_id
        / f"seed_{seed:03d}"
        / "run_summary.json"
    )


def run_one_seed(config_path: str, manifest_path: str, seed: int) -> int:
    cmd = [
        sys.executable,
        "-m",
        "scripts.training.run_experiment",
        "--config",
        config_path,
        "--manifest",
        manifest_path,
        "--seed",
        str(seed),
    ]

    print(f"\n=== Running seed {seed} ===")
    print(" ".join(cmd))

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main() -> None:
    args = parse_args()

    config = load_yaml(args.config)
    exp_id = get_exp_id(config)
    seeds = get_seed_list(config)

    print(f"Experiment: {exp_id}")
    print(f"Seeds: {seeds}")
    print(f"Force rerun: {args.force}")

    failed_seeds: list[int] = []

    for seed in seeds:
        summary_path = seed_run_summary_path(exp_id, seed)

        if summary_path.exists() and not args.force:
            print(f"\n=== Skipping seed {seed} (already completed) ===")
            print(f"Found: {summary_path}")
            continue

        return_code = run_one_seed(
            config_path=args.config,
            manifest_path=args.manifest,
            seed=seed,
        )

        if return_code != 0:
            failed_seeds.append(seed)
            print(f"\nSeed {seed} failed with exit code {return_code}")

    print("\n=== Multiseed run complete ===")
    if failed_seeds:
        print(f"Failed seeds: {failed_seeds}")
        raise SystemExit(1)

    print("All requested seeds completed successfully.")


if __name__ == "__main__":
    main()