from __future__ import annotations

import faiss
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer


def get_embedding_text(
    example: dict,
    instruction_only: bool = False,
) -> str:
    """
    Build the text used for semantic embeddings.

    This reproduces the current Group 1 notebook protocol:
    - instruction only when requested
    - otherwise instruction + context when context is available
    """
    instruction = example["instruction"]

    if instruction_only:
        return instruction

    context = example.get("context")

    if context:
        return f"{instruction} {context}"

    return instruction


def build_semantic_index(
    train_dataset: Dataset,
    embedding_model_name: str,
    instruction_only: bool = False,
) -> tuple[np.ndarray, faiss.IndexFlatIP]:
    """
    Encode the training examples and build the FAISS index used for
    semantic batching.

    Embeddings are L2-normalized and IndexFlatIP is used so that inner
    product corresponds to cosine similarity.
    """
    embedding_texts = [
        get_embedding_text(
            example,
            instruction_only=instruction_only,
        )
        for example in train_dataset
    ]

    embedder = SentenceTransformer(embedding_model_name)

    embeddings = embedder.encode(
        embedding_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return embeddings, index