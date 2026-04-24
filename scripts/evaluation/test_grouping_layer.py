from __future__ import annotations

from pathlib import Path

from src.batching.grouping import (
    build_anchor_to_group_map,
    get_grouped_neighbors_for_processed_anchor,
    summarize_group_map,
)
from src.batching.index_loader import load_semantic_index
from src.data.alignment import build_alignment, validate_alignment
from src.data.processed_loader import load_processed_dataset
from src.data.raw_loader import load_raw_dataset


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
    print("ALIGNMENT:", validate_alignment(alignment))

    semantic_index = load_semantic_index(semantic_index_path)
    print("SEMANTIC INDEX N_ROWS:", semantic_index.n_rows)
    print("SEMANTIC INDEX TOP_K:", semantic_index.top_k)

    grouped_view = get_grouped_neighbors_for_processed_anchor(
        anchor_processed_idx=0,
        alignment=alignment,
        semantic_index=semantic_index,
        max_group_size=8,
        include_anchor=True,
    )

    print("\nSINGLE ANCHOR VIEW")
    print("anchor_processed_idx:", grouped_view.anchor_processed_idx)
    print("anchor_raw_idx:", grouped_view.anchor_raw_idx)
    print("neighbor_processed_indices:", grouped_view.neighbor_processed_indices)
    print("neighbor_raw_indices:", grouped_view.neighbor_raw_indices)
    print("dropped_neighbor_ids (first 20):", grouped_view.dropped_neighbor_ids[:20])

    anchor_to_group = build_anchor_to_group_map(
        alignment=alignment,
        semantic_index=semantic_index,
        max_group_size=8,
        include_anchor=True,
    )
    print("\nGROUP MAP SUMMARY")
    print(summarize_group_map(anchor_to_group))


if __name__ == "__main__":
    main()