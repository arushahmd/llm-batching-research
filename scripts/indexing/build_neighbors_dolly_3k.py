# scripts/build_neighbors_dolly_3k.py

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "semantic_groups"
    / "dolly_3k__all-MiniLM-L6-v2"
)

TOP_K = 8


def main() -> None:
    embeddings_path = INDEX_DIR / "embeddings.npy"
    faiss_path = INDEX_DIR / "faiss.index"
    meta_path = INDEX_DIR / "meta.json"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")
    if not faiss_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {faiss_path}")

    embeddings = np.load(embeddings_path).astype("float32")
    index = faiss.read_index(str(faiss_path))

    print(f"Searching top-{TOP_K} neighbors for {len(embeddings)} rows...")
    scores, indices = index.search(embeddings, TOP_K)

    np.save(INDEX_DIR / "neighbors_topk_idx.npy", indices)
    np.save(INDEX_DIR / "neighbors_topk_scores.npy", scores)

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    meta["top_k"] = TOP_K
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Indices shape:", indices.shape)
    print("Scores shape:", scores.shape)


if __name__ == "__main__":
    main()