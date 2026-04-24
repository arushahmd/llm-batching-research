# src/batching/samplers.py

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator, Literal


SamplerMode = Literal["random", "grouped"]


@dataclass(frozen=True)
class SamplerStats:
    sampler_mode: str
    n_examples: int
    batch_size: int
    n_batches: int


class RandomBatchSampler:
    """
    Yields batches of processed train indices in random order.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        indices = list(range(self.dataset_size))
        rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)

    def stats(self) -> SamplerStats:
        return SamplerStats(
            sampler_mode="random",
            n_examples=self.dataset_size,
            batch_size=self.batch_size,
            n_batches=len(self),
        )


class GroupedBatchSampler:
    """
    Yields grouped batches of processed train indices using a precomputed anchor->group map.

    Strategy:
    - shuffle anchors
    - for each unseen anchor, emit its group
    - mark emitted members as seen
    - continue until all examples are covered
    - optionally pad underfilled batches from remaining unseen items
    """

    def __init__(
        self,
        anchor_to_group: dict[int, list[int]],
        dataset_size: int,
        batch_size: int,
        seed: int,
        drop_last: bool = False,
        pad_incomplete_batches: bool = True,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not anchor_to_group:
            raise ValueError("anchor_to_group must not be empty")

        self.anchor_to_group = anchor_to_group
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.pad_incomplete_batches = pad_incomplete_batches

        self._validate()

    def _validate(self) -> None:
        for anchor, group in self.anchor_to_group.items():
            if anchor < 0 or anchor >= self.dataset_size:
                raise ValueError(f"Invalid anchor index: {anchor}")

            if not group:
                raise ValueError(f"Anchor {anchor} has empty group")

            for idx in group:
                if idx < 0 or idx >= self.dataset_size:
                    raise ValueError(
                        f"Group for anchor {anchor} contains invalid processed index {idx}"
                    )

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        anchors = list(self.anchor_to_group.keys())
        rng.shuffle(anchors)

        seen: set[int] = set()

        for anchor in anchors:
            if anchor in seen:
                continue

            group = self.anchor_to_group[anchor]

            batch: list[int] = []
            for idx in group:
                if idx not in seen:
                    batch.append(idx)
                    seen.add(idx)
                if len(batch) == self.batch_size:
                    break

            if len(batch) < self.batch_size and self.pad_incomplete_batches:
                remaining = [i for i in range(self.dataset_size) if i not in seen]
                rng.shuffle(remaining)
                need = self.batch_size - len(batch)
                fillers = remaining[:need]
                batch.extend(fillers)
                seen.update(fillers)

            if self.drop_last and len(batch) < self.batch_size:
                continue

            if batch:
                yield batch

        # Safety pass: emit any leftover unseen examples
        leftovers = [i for i in range(self.dataset_size) if i not in seen]
        if leftovers:
            rng.shuffle(leftovers)
            for start in range(0, len(leftovers), self.batch_size):
                batch = leftovers[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)

    def stats(self) -> SamplerStats:
        return SamplerStats(
            sampler_mode="grouped",
            n_examples=self.dataset_size,
            batch_size=self.batch_size,
            n_batches=len(self),
        )


def make_batch_sampler(
    sampler_mode: SamplerMode,
    dataset_size: int,
    batch_size: int,
    seed: int,
    drop_last: bool = False,
    anchor_to_group: dict[int, list[int]] | None = None,
) -> RandomBatchSampler | GroupedBatchSampler:
    if sampler_mode == "random":
        return RandomBatchSampler(
            dataset_size=dataset_size,
            batch_size=batch_size,
            seed=seed,
            drop_last=drop_last,
        )

    if sampler_mode == "grouped":
        if anchor_to_group is None:
            raise ValueError("anchor_to_group is required when sampler_mode='grouped'")
        return GroupedBatchSampler(
            anchor_to_group=anchor_to_group,
            dataset_size=dataset_size,
            batch_size=batch_size,
            seed=seed,
            drop_last=drop_last,
            pad_incomplete_batches=True,
        )

    raise ValueError(f"Unsupported sampler_mode: {sampler_mode}")