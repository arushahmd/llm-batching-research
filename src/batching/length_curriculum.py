from __future__ import annotations

from typing import Any


VALID_DIRECTIONS = {"easy_to_hard", "hard_to_easy"}


def compute_input_lengths(
    train_dataset: Any,
    tokenizer: Any,
    instruction_only: bool = False,
) -> list[int]:
    """Compute full tokenizer-output lengths, matching executed Group 2."""
    lengths: list[int] = []

    for i in range(len(train_dataset)):
        example = train_dataset[i]
        text = (
            example["instruction"]
            if instruction_only
            else example["input_text"]
        )
        lengths.append(len(tokenizer(text)["input_ids"]))

    return lengths


def build_length_order(
    lengths: list[int],
    direction: str,
    n_batches: int,
    batch_size: int,
) -> list[int]:
    """Build the fully sorted, cycling fixed order used by Group 2."""
    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"Unknown direction {direction!r}; "
            f"expected {sorted(VALID_DIRECTIONS)}."
        )

    if n_batches < 0:
        raise ValueError("n_batches must be non-negative.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    total_needed = n_batches * batch_size

    if total_needed == 0:
        return []

    if not lengths:
        raise ValueError("lengths cannot be empty for a non-empty order.")

    ascending = sorted(
        range(len(lengths)),
        key=lambda i: lengths[i],
    )
    base = (
        ascending
        if direction == "easy_to_hard"
        else list(reversed(ascending))
    )
    repeats = (total_needed + len(base) - 1) // len(base)

    return (base * repeats)[:total_needed]
