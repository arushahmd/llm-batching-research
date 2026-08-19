from __future__ import annotations

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase


def format_dolly_example(example: dict) -> dict[str, str]:
    """Convert a Dolly example into the input/target format used in Group 1."""
    instruction = example["instruction"]
    context = example.get("context")
    response = example["response"]

    if context:
        input_text = f"Instruction: {instruction}\nContext: {context}"
    else:
        input_text = f"Instruction: {instruction}"

    return {
        "input_text": input_text,
        "target_text": response,
    }


def load_dolly_split(
    dataset_name: str,
    subset_size: int,
    shuffle_seed: int,
    eval_ratio: float,
    split_seed: int,
) -> tuple[Dataset, Dataset]:
    """
    Load, shuffle, subset, format, and split Dolly.

    The same deterministic data split is reused across all strategies and
    experiment seeds so that batching strategy is the primary varying factor.
    """
    dataset = load_dataset(dataset_name, split="train")

    if subset_size > len(dataset):
        raise ValueError(
            f"subset_size={subset_size} exceeds dataset size={len(dataset)}"
        )

    dataset = (
        dataset
        .shuffle(seed=shuffle_seed)
        .select(range(subset_size))
        .map(format_dolly_example)
    )

    split = dataset.train_test_split(
        test_size=eval_ratio,
        seed=split_seed,
    )

    return split["train"], split["test"]


def tokenize_dolly_split(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int,
    max_target_length: int,
) -> tuple[Dataset, Dataset]:
    """
    Tokenize Group 1 train/evaluation datasets.

    Note:
        Label padding behavior intentionally reproduces the currently completed
        Group 1 notebooks. Any change to ignored label padding should be treated
        as a protocol change and evaluated in a separate rerun.
    """

    def preprocess(example: dict) -> dict:
        model_inputs = tokenizer(
            example["input_text"],
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
        )

        labels = tokenizer(
            text_target=example["target_text"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length",
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tokenized = train_dataset.map(
        preprocess,
        remove_columns=train_dataset.column_names,
    )

    eval_tokenized = eval_dataset.map(
        preprocess,
        remove_columns=eval_dataset.column_names,
    )

    return train_tokenized, eval_tokenized