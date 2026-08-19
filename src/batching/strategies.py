from __future__ import annotations

import numpy as np


def build_random_order(
    n_examples: int,
    batch_size: int,
    n_batches: int,
    seed: int,
) -> list[int]:
    """
    Build the random batching order used in Group 1.

    Each physical batch is sampled independently from the full training
    dataset without replacement within that batch. Examples may therefore
    appear again in later batches.
    """
    if batch_size > n_examples:
        raise ValueError(
            f"batch_size={batch_size} cannot exceed n_examples={n_examples}"
        )

    rng = np.random.RandomState(seed)
    order: list[int] = []

    for _ in range(n_batches):
        batch = rng.choice(
            n_examples,
            size=batch_size,
            replace=False,
        )
        order.extend(batch.tolist())

    return order


def build_grouped_order(
    embeddings: np.ndarray,
    index,
    n_examples: int,
    batch_size: int,
    top_k: int,
    n_batches: int,
    seed: int,
) -> list[int]:
    """
    Build the semantically grouped batching order used in Group 1.

    Each batch consists of:
        1. one anchor example;
        2. up to batch_size - 1 examples sampled from the anchor's
           top-k semantic-neighbor pool.

    Anchors are drawn from a shuffled pool. Once every training example has
    served as an anchor, the pool is reshuffled and reused.

    If too few semantic neighbors are available, remaining positions are
    filled by random examples, matching the current notebook protocol.
    """
    if batch_size > n_examples:
        raise ValueError(
            f"batch_size={batch_size} cannot exceed n_examples={n_examples}"
        )

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    rng = np.random.RandomState(seed)

    order: list[int] = []

    anchor_pool = list(range(n_examples))
    rng.shuffle(anchor_pool)

    pool_idx = 0

    for _ in range(n_batches):
        if pool_idx >= len(anchor_pool):
            rng.shuffle(anchor_pool)
            pool_idx = 0

        anchor = anchor_pool[pool_idx]
        pool_idx += 1

        query_vec = embeddings[anchor : anchor + 1]

        # +1 because the anchor itself is normally returned by FAISS.
        _, neighbor_ids = index.search(
            query_vec,
            top_k + 1,
        )

        neighbor_ids = [
            i
            for i in neighbor_ids[0]
            if i != anchor
        ][:top_k]

        chosen = rng.choice(
            neighbor_ids,
            size=min(batch_size - 1, len(neighbor_ids)),
            replace=False,
        )

        batch = [anchor] + chosen.tolist()

        # Preserve the fallback behavior of the current experiment notebook.
        while len(batch) < batch_size:
            batch.append(int(rng.choice(n_examples)))

        order.extend(batch)

    return order


def build_curriculum_order(
    strategy: str,
    embeddings: np.ndarray,
    index,
    n_examples: int,
    batch_size: int,
    top_k: int,
    n_batches: int,
    seed: int,
) -> list[int]:
    """
    Build one of the two-phase curriculum schedules used in Group 1.

    grouped_to_random:
        first half grouped, second half random

    random_to_grouped:
        first half random, second half grouped

    The second phase uses seed + 1000, reproducing the current experiment
    implementation.
    """
    half = n_batches // 2

    if strategy == "grouped_to_random":
        first = build_grouped_order(
            embeddings=embeddings,
            index=index,
            n_examples=n_examples,
            batch_size=batch_size,
            top_k=top_k,
            n_batches=half,
            seed=seed,
        )

        second = build_random_order(
            n_examples=n_examples,
            batch_size=batch_size,
            n_batches=n_batches - half,
            seed=seed + 1000,
        )

    elif strategy == "random_to_grouped":
        first = build_random_order(
            n_examples=n_examples,
            batch_size=batch_size,
            n_batches=half,
            seed=seed,
        )

        second = build_grouped_order(
            embeddings=embeddings,
            index=index,
            n_examples=n_examples,
            batch_size=batch_size,
            top_k=top_k,
            n_batches=n_batches - half,
            seed=seed + 1000,
        )

    else:
        raise ValueError(
            f"Unsupported curriculum strategy: {strategy}"
        )

    return first + second


def build_batch_order(
    strategy: str,
    embeddings: np.ndarray,
    index,
    n_examples: int,
    batch_size: int,
    top_k: int,
    n_batches: int,
    seed: int,
) -> list[int]:
    """
    Build the complete fixed training order for a Group 1 strategy.
    """
    if strategy == "random":
        return build_random_order(
            n_examples=n_examples,
            batch_size=batch_size,
            n_batches=n_batches,
            seed=seed,
        )

    if strategy == "grouped":
        return build_grouped_order(
            embeddings=embeddings,
            index=index,
            n_examples=n_examples,
            batch_size=batch_size,
            top_k=top_k,
            n_batches=n_batches,
            seed=seed,
        )

    if strategy in {
        "grouped_to_random",
        "random_to_grouped",
    }:
        return build_curriculum_order(
            strategy=strategy,
            embeddings=embeddings,
            index=index,
            n_examples=n_examples,
            batch_size=batch_size,
            top_k=top_k,
            n_batches=n_batches,
            seed=seed,
        )

    raise ValueError(f"Unsupported batching strategy: {strategy}")