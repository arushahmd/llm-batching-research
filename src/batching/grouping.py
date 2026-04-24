# src/batching/grouping.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.batching.index_loader import SemanticIndexBundle
from src.data.alignment import DatasetAlignment


@dataclass(frozen=True)
class GroupedNeighborView:
    anchor_processed_idx: int
    anchor_raw_idx: int
    neighbor_processed_indices: list[int]
    neighbor_raw_indices: list[int]
    dropped_neighbor_ids: list[int]


def build_train_raw_idx_set(alignment: DatasetAlignment) -> set[int]:
    return set(alignment.raw_idx_to_processed_idx.keys())


def semantic_row_to_raw_idx(
    semantic_row_idx: int,
    semantic_index: SemanticIndexBundle,
) -> int:
    if semantic_row_idx < 0 or semantic_row_idx >= len(semantic_index.id_map):
        raise IndexError(
            f"semantic_row_idx={semantic_row_idx} out of range for id_map of size {len(semantic_index.id_map)}"
        )
    return semantic_index.id_map[semantic_row_idx]


def get_train_eligible_neighbor_raw_indices(
    anchor_raw_idx: int,
    semantic_index: SemanticIndexBundle,
    train_raw_idx_set: set[int],
    include_anchor: bool = True,
) -> tuple[list[int], list[int]]:
    """
    Returns:
        kept_raw_indices, dropped_raw_indices
    """
    if anchor_raw_idx < 0 or anchor_raw_idx >= semantic_index.n_rows:
        raise IndexError(
            f"anchor_raw_idx={anchor_raw_idx} out of semantic index range [0, {semantic_index.n_rows})"
        )

    neighbor_semantic_rows = semantic_index.neighbors_idx[anchor_raw_idx].tolist()

    kept_raw_indices: list[int] = []
    dropped_raw_indices: list[int] = []

    for semantic_row in neighbor_semantic_rows:
        neighbor_raw_idx = semantic_row_to_raw_idx(semantic_row, semantic_index)

        if not include_anchor and neighbor_raw_idx == anchor_raw_idx:
            continue

        if neighbor_raw_idx in train_raw_idx_set:
            kept_raw_indices.append(neighbor_raw_idx)
        else:
            dropped_raw_indices.append(neighbor_raw_idx)

    return kept_raw_indices, dropped_raw_indices


def get_grouped_neighbors_for_processed_anchor(
    anchor_processed_idx: int,
    alignment: DatasetAlignment,
    semantic_index: SemanticIndexBundle,
    max_group_size: int,
    include_anchor: bool = True,
) -> GroupedNeighborView:
    """
    Build a train-valid semantic group for a processed train anchor.

    Assumption:
    - semantic index id_map values align to raw_idx space
    - processed train rows map to raw_idx via alignment
    """
    anchor_raw_idx = alignment.get_raw_idx_from_processed_idx(anchor_processed_idx)
    train_raw_idx_set = build_train_raw_idx_set(alignment)

    kept_raw_indices, dropped_raw_indices = get_train_eligible_neighbor_raw_indices(
        anchor_raw_idx=anchor_raw_idx,
        semantic_index=semantic_index,
        train_raw_idx_set=train_raw_idx_set,
        include_anchor=include_anchor,
    )

    trimmed_raw_indices = kept_raw_indices[:max_group_size]
    neighbor_processed_indices = [
        alignment.get_processed_idx_from_raw_idx(raw_idx)
        for raw_idx in trimmed_raw_indices
    ]

    return GroupedNeighborView(
        anchor_processed_idx=anchor_processed_idx,
        anchor_raw_idx=anchor_raw_idx,
        neighbor_processed_indices=neighbor_processed_indices,
        neighbor_raw_indices=trimmed_raw_indices,
        dropped_neighbor_ids=dropped_raw_indices,
    )


def build_anchor_to_group_map(
    alignment: DatasetAlignment,
    semantic_index: SemanticIndexBundle,
    max_group_size: int,
    include_anchor: bool = True,
) -> dict[int, list[int]]:
    """
    For every processed train row, build a semantic group in processed-index space.
    """
    anchor_to_group: dict[int, list[int]] = {}

    for anchor_processed_idx in range(len(alignment.processed_train_ds)):
        grouped_view = get_grouped_neighbors_for_processed_anchor(
            anchor_processed_idx=anchor_processed_idx,
            alignment=alignment,
            semantic_index=semantic_index,
            max_group_size=max_group_size,
            include_anchor=include_anchor,
        )
        anchor_to_group[anchor_processed_idx] = grouped_view.neighbor_processed_indices

    return anchor_to_group


def summarize_group_map(anchor_to_group: dict[int, list[int]]) -> dict[str, Any]:
    group_sizes = [len(v) for v in anchor_to_group.values()]
    if not group_sizes:
        return {
            "n_anchors": 0,
            "min_group_size": 0,
            "max_group_size": 0,
            "mean_group_size": 0.0,
        }

    return {
        "n_anchors": len(group_sizes),
        "min_group_size": min(group_sizes),
        "max_group_size": max(group_sizes),
        "mean_group_size": sum(group_sizes) / len(group_sizes),
    }