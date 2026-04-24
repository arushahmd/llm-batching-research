from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    index_dir = (
        project_root
        / "data"
        / "indexes"
        / "semantic_groups"
        / "dolly_small_1k__all-MiniLM-L6-v2"
    )

    embeddings_path = index_dir / "embeddings.npy"
    neighbors_idx_path = index_dir / "neighbors_topk_idx.npy"
    neighbors_scores_path = index_dir / "neighbors_topk_scores.npy"
    id_map_path = index_dir / "id_map.json"
    meta_path = index_dir / "meta.json"

    embeddings = np.load(embeddings_path)
    neighbors_idx = np.load(neighbors_idx_path)
    neighbors_scores = np.load(neighbors_scores_path)

    with open(id_map_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("INDEX DIR:", index_dir)
    print("embeddings shape:", embeddings.shape)
    print("neighbors_idx shape:", neighbors_idx.shape)
    print("neighbors_scores shape:", neighbors_scores.shape)
    print("id_map type:", type(id_map).__name__)
    if isinstance(id_map, dict):
        print("id_map keys:", list(id_map.keys())[:20])
        for k in list(id_map.keys())[:3]:
            print(f"id_map[{k!r}] =", id_map[k])
    elif isinstance(id_map, list):
        print("id_map length:", len(id_map))
        print("id_map first 10:", id_map[:10])

    print("meta keys:", list(meta.keys()))
    print("meta:", meta)

    print("\nSample neighbors:")
    for row_idx in range(min(3, len(neighbors_idx))):
        print(f"row {row_idx}:")
        print("  neighbor idx:", neighbors_idx[row_idx][:10].tolist())
        print("  scores      :", neighbors_scores[row_idx][:10].tolist())


if __name__ == "__main__":
    main()