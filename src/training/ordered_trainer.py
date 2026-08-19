from __future__ import annotations

import torch
from transformers import Seq2SeqTrainer


class FixedOrderSampler(torch.utils.data.Sampler):
    """
    Sampler that yields dataset indices in one predefined order.

    Group 1 constructs the complete physical-batch sequence before training.
    This sampler ensures the Trainer consumes examples in exactly that order.
    """

    def __init__(self, indices: list[int]):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class OrderedTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer that uses a predefined training-example order.

    The fixed order is produced by one of the Group 1 batching strategies:
    random, grouped, grouped_to_random, or random_to_grouped.
    """

    def __init__(
        self,
        *args,
        fixed_order: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fixed_order = fixed_order

    def get_train_dataloader(self):
        if self.fixed_order is None:
            raise ValueError(
                "OrderedTrainer requires fixed_order to be provided."
            )

        sampler = FixedOrderSampler(self.fixed_order)

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=True,
        )