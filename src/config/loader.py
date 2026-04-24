# src/config/loader.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REQUIRED_EXPERIMENT_TOP_LEVEL_KEYS = {
    "experiment",
    "assets",
    "dataset",
    "multiseed",
    "lora",
    "training",
    "grouping",
    "phases",
}


@dataclass(frozen=True)
class ResolvedPaths:
    project_root: Path
    experiments_root: Path
    base_model_path: Path
    embedding_model_path: Path
    raw_dataset_path: Path
    processed_dataset_path: Path
    semantic_index_path: Path


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML file '{path}' to contain a dictionary at top level")

    return data


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON file '{path}' to contain a dictionary at top level")

    return data


def validate_experiment_config(config: dict[str, Any]) -> None:
    missing = REQUIRED_EXPERIMENT_TOP_LEVEL_KEYS - set(config.keys())
    if missing:
        raise ValueError(
            f"Experiment config missing required top-level keys: {sorted(missing)}"
        )

    if not isinstance(config["phases"], list) or not config["phases"]:
        raise ValueError("Experiment config 'phases' must be a non-empty list")

    if not isinstance(config["dataset"], dict) or "name" not in config["dataset"]:
        raise ValueError("Experiment config must contain dataset.name")

    for idx, phase in enumerate(config["phases"]):
        if not isinstance(phase, dict):
            raise ValueError(f"Phase at index {idx} must be a dictionary")
        required_phase_keys = {"phase_name", "sampler_mode", "epochs", "learning_rate"}
        missing_phase_keys = required_phase_keys - set(phase.keys())
        if missing_phase_keys:
            raise ValueError(
                f"Phase at index {idx} missing required keys: {sorted(missing_phase_keys)}"
            )


def _resolve_dataset_variant_paths(
    manifest: dict[str, Any],
    dataset_name: str,
) -> tuple[Path, Path, Path]:
    dataset_variants = manifest.get("dataset_variants", {})

    if dataset_name in dataset_variants:
        variant = dataset_variants[dataset_name]
        return (
            Path(variant["raw_dataset_path"]),
            Path(variant["processed_dataset_path"]),
            Path(variant["semantic_index_path"]),
        )

    assets = manifest["assets"]
    return (
        Path(assets["raw_dataset"]["path"]),
        Path(assets["processed_dataset"]["path"]),
        Path(assets["semantic_index"]["path"]),
    )


def resolve_paths_from_manifest(
    manifest: dict[str, Any],
    experiment_config: dict[str, Any],
) -> ResolvedPaths:
    root = Path(manifest["root"])
    assets = manifest["assets"]
    paths = manifest["paths"]

    dataset_name = experiment_config["dataset"]["name"]

    raw_dataset_path, processed_dataset_path, semantic_index_path = _resolve_dataset_variant_paths(
        manifest=manifest,
        dataset_name=dataset_name,
    )

    return ResolvedPaths(
        project_root=root,
        experiments_root=Path(paths["experiments"]),
        base_model_path=Path(assets["base_model"]["path"]),
        embedding_model_path=Path(assets["embedding_model"]["path"]),
        raw_dataset_path=raw_dataset_path,
        processed_dataset_path=processed_dataset_path,
        semantic_index_path=semantic_index_path,
    )


def load_experiment_and_manifest(
    experiment_config_path: str | Path,
    project_manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], ResolvedPaths]:
    exp_config = load_yaml(experiment_config_path)
    validate_experiment_config(exp_config)

    project_manifest = load_json(project_manifest_path)
    resolved_paths = resolve_paths_from_manifest(project_manifest, exp_config)

    return exp_config, project_manifest, resolved_paths