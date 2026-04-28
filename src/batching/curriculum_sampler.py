# src/batching/curriculum_sampler.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import torch
from torch.utils.data import Sampler


CurriculumMode = Literal[
    "easy_to_hard_length",
    "hard_to_easy_length",
    "random",
]


@dataclass(frozen=True)
class CurriculumBatchingConfig:
    mode: CurriculumMode
    batch_size: int
    seed: int = 42
    label_pad_token_id: int = -100


class CurriculumBatchSampler(Sampler[list[int]]):
    """
    Batch sampler for curriculum-based instruction ordering.

    Supported modes:
    - easy_to_hard_length: shortest examples first
    - hard_to_easy_length: longest examples first
    - random: shuffled baseline
    """

    def __init__(
        self,
        dataset,
        config: CurriculumBatchingConfig,
    ) -> None:
        self.dataset = dataset
        self.config = config

        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.indices = self._build_ordered_indices()

    def _sample_length(self, idx: int) -> int:
        sample = self.dataset[idx]

        input_ids = sample.get("input_ids")
        labels = sample.get("labels")

        input_len = len(input_ids) if input_ids is not None else 0

        if labels is None:
            label_len = 0
        else:
            label_len = sum(
                1 for token_id in labels
                if token_id != self.config.label_pad_token_id
            )

        return input_len + label_len

    def _build_ordered_indices(self) -> list[int]:
        indices = list(range(len(self.dataset)))

        if self.config.mode == "random":
            generator = torch.Generator()
            generator.manual_seed(self.config.seed)
            perm = torch.randperm(len(indices), generator=generator).tolist()
            return [indices[i] for i in perm]

        lengths = [(idx, self._sample_length(idx)) for idx in indices]

        if self.config.mode == "easy_to_hard_length":
            return [idx for idx, _ in sorted(lengths, key=lambda x: x[1])]

        if self.config.mode == "hard_to_easy_length":
            return [idx for idx, _ in sorted(lengths, key=lambda x: x[1], reverse=True)]

        raise ValueError(f"Unsupported curriculum mode: {self.config.mode}")

    def __iter__(self) -> Iterator[list[int]]:
        batch = []

        for idx in self.indices:
            batch.append(idx)

            if len(batch) == self.config.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def __len__(self) -> int:
        n = len(self.indices)
        return (n + self.config.batch_size - 1) // self.config.batch_size