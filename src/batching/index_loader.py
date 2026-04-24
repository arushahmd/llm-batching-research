# src/batching/index_loader.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SemanticIndexBundle:
    index_dir: Path
    embeddings: np.ndarray
    neighbors_idx: np.ndarray
    neighbors_scores: np.ndarray
    id_map: list[int]
    meta: dict[str, Any]

    @property
    def n_rows(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def top_k(self) -> int:
        return int(self.neighbors_idx.shape[1])


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_semantic_index(index_dir: str | Path) -> SemanticIndexBundle:
    index_dir = Path(index_dir)

    if not index_dir.exists():
        raise FileNotFoundError(f"Semantic index directory does not exist: {index_dir}")

    embeddings_path = index_dir / "embeddings.npy"
    neighbors_idx_path = index_dir / "neighbors_topk_idx.npy"
    neighbors_scores_path = index_dir / "neighbors_topk_scores.npy"
    id_map_path = index_dir / "id_map.json"
    meta_path = index_dir / "meta.json"

    required_paths = [
        embeddings_path,
        neighbors_idx_path,
        neighbors_scores_path,
        id_map_path,
        meta_path,
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Semantic index directory is missing required files:\n" + "\n".join(missing)
        )

    embeddings = np.load(embeddings_path)
    neighbors_idx = np.load(neighbors_idx_path)
    neighbors_scores = np.load(neighbors_scores_path)

    id_map_payload = _load_json(id_map_path)
    meta = _load_json(meta_path)

    if "id_map" not in id_map_payload:
        raise ValueError(f"id_map.json at '{id_map_path}' must contain key 'id_map'")

    raw_id_map = id_map_payload["id_map"]
    if not isinstance(raw_id_map, list):
        raise TypeError("id_map.json['id_map'] must be a list")

    try:
        id_map = [int(x) for x in raw_id_map]
    except Exception as exc:
        raise ValueError("Failed to parse id_map values as integers") from exc

    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings to be 2D, got shape={embeddings.shape}")

    if neighbors_idx.ndim != 2 or neighbors_scores.ndim != 2:
        raise ValueError(
            f"Expected neighbors arrays to be 2D, got idx={neighbors_idx.shape}, "
            f"scores={neighbors_scores.shape}"
        )

    n_rows = embeddings.shape[0]
    if neighbors_idx.shape[0] != n_rows or neighbors_scores.shape[0] != n_rows:
        raise ValueError(
            "Semantic index row mismatch: "
            f"embeddings={embeddings.shape}, "
            f"neighbors_idx={neighbors_idx.shape}, "
            f"neighbors_scores={neighbors_scores.shape}"
        )

    if len(id_map) != n_rows:
        raise ValueError(
            f"id_map length mismatch: len(id_map)={len(id_map)} but n_rows={n_rows}"
        )

    return SemanticIndexBundle(
        index_dir=index_dir,
        embeddings=embeddings,
        neighbors_idx=neighbors_idx,
        neighbors_scores=neighbors_scores,
        id_map=id_map,
        meta=meta,
    )