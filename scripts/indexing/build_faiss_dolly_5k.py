# scripts/build_faiss_dolly_5k.py

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "semantic_groups"
    / "dolly_5k__all-MiniLM-L6-v2"
)


def main() -> None:
    embeddings_path = INDEX_DIR / "embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")

    embeddings = np.load(embeddings_path).astype("float32")

    print(f"Loaded embeddings from: {embeddings_path}")
    print("Embeddings shape:", embeddings.shape)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss_path = INDEX_DIR / "faiss.index"
    faiss.write_index(index, str(faiss_path))

    print(f"Saved FAISS index to: {faiss_path}")
    print("Index ntotal:", index.ntotal)


if __name__ == "__main__":
    main()