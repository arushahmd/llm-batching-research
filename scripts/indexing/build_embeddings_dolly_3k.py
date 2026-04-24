# scripts/build_embeddings_dolly_3k.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from sentence_transformers import SentenceTransformer

from src.data.raw_loader import build_embedding_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "dolly_3k"
MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "indexes"
    / "semantic_groups"
    / "dolly_3k__all-MiniLM-L6-v2"
)

TEXT_MODE = "instruction_plus_context"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_raw_train_dataset(path: Path):
    ds = load_from_disk(str(path))

    if isinstance(ds, DatasetDict):
        if "train" not in ds:
            raise ValueError(f"DatasetDict at {path} does not contain a 'train' split.")
        return ds["train"]

    if isinstance(ds, Dataset):
        return ds

    raise TypeError(f"Expected Dataset or DatasetDict at {path}, got: {type(ds).__name__}")


def main() -> None:
    raw_train = load_raw_train_dataset(RAW_PATH)

    print(f"Loaded raw dataset from: {RAW_PATH}")
    print(f"Train rows: {len(raw_train)}")

    print(f"Loading embedding model from: {MODEL_PATH}")
    model = SentenceTransformer(str(MODEL_PATH))

    print("Building embedding texts...")
    texts = [
        build_embedding_text(sample, mode=TEXT_MODE)
        for sample in raw_train
    ]

    print(f"Encoding {len(texts)} embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)

    id_map_payload = {"id_map": list(range(len(raw_train)))}
    with open(OUTPUT_DIR / "id_map.json", "w", encoding="utf-8") as f:
        json.dump(id_map_payload, f, indent=2)

    meta = {
        "dataset_name": "dolly_3k",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "text_mode": TEXT_MODE,
        "n_rows": len(raw_train),
        "embedding_dim": int(embeddings.shape[1]),
        "normalized": True,
    }
    with open(OUTPUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved embeddings to: {OUTPUT_DIR / 'embeddings.npy'}")
    print(f"Saved id_map to: {OUTPUT_DIR / 'id_map.json'}")
    print(f"Saved meta to: {OUTPUT_DIR / 'meta.json'}")
    print("Embeddings shape:", embeddings.shape)


if __name__ == "__main__":
    main()