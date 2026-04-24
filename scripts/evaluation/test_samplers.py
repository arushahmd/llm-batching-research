from __future__ import annotations

from pathlib import Path

from src.batching.grouping import build_anchor_to_group_map
from src.batching.index_loader import load_semantic_index
from src.batching.samplers import make_batch_sampler
from src.data.alignment import build_alignment
from src.data.processed_loader import load_processed_dataset
from src.data.raw_loader import load_raw_dataset


def summarize_batches(name: str, batches: list[list[int]], dataset_size: int) -> None:
    flat = [idx for batch in batches for idx in batch]
    unique = set(flat)

    print(f"\n{name}")
    print(f"n_batches: {len(batches)}")
    print(f"first 3 batches: {batches[:3]}")
    print(f"n_items_emitted: {len(flat)}")
    print(f"n_unique_items: {len(unique)}")
    print(f"coverage_ratio: {len(unique) / dataset_size:.4f}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    raw_path = project_root / "data" / "raw" / "dolly_15k"
    processed_path = (
        project_root
        / "data"
        / "processed"
        / "dolly_small_1k__flan-t5-small__tok_with_rawidx_v1"
    )
    semantic_index_path = (
        project_root
        / "data"
        / "indexes"
        / "semantic_groups"
        / "dolly_small_1k__all-MiniLM-L6-v2"
    )

    raw_dataset = load_raw_dataset(raw_path)
    processed_dataset = load_processed_dataset(processed_path)

    alignment = build_alignment(
        processed_train_ds=processed_dataset["train"],
        raw_train_ds=raw_dataset["train"],
    )
    semantic_index = load_semantic_index(semantic_index_path)

    dataset_size = len(processed_dataset["train"])
    batch_size = 8
    seed = 42

    anchor_to_group = build_anchor_to_group_map(
        alignment=alignment,
        semantic_index=semantic_index,
        max_group_size=batch_size,
        include_anchor=True,
    )

    random_sampler = make_batch_sampler(
        sampler_mode="random",
        dataset_size=dataset_size,
        batch_size=batch_size,
        seed=seed,
        drop_last=False,
    )
    grouped_sampler = make_batch_sampler(
        sampler_mode="grouped",
        dataset_size=dataset_size,
        batch_size=batch_size,
        seed=seed,
        drop_last=False,
        anchor_to_group=anchor_to_group,
    )

    random_batches = list(random_sampler)
    grouped_batches = list(grouped_sampler)

    summarize_batches("RANDOM", random_batches, dataset_size)
    summarize_batches("GROUPED", grouped_batches, dataset_size)


if __name__ == "__main__":
    main()