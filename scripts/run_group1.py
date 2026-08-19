from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
)

# Allow running:
# python scripts/run_group1.py
REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.batching.semantic import build_semantic_index
from src.batching.strategies import build_batch_order
from src.data.dolly import (
    load_dolly_split,
    tokenize_dolly_split,
)
from src.evaluation.generation import evaluate_generation
from src.training.model import build_lora_seq2seq_model
from src.training.ordered_trainer import OrderedTrainer
from src.utils.reproducibility import set_experiment_seed


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_result(
    result_path: Path,
    result: dict,
) -> None:
    """
    Save results incrementally.

    Existing strategy/seed results are replaced so rerunning one experiment
    does not create duplicate entries.
    """
    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as file:
            results = json.load(file)
    else:
        results = []

    results = [
        existing
        for existing in results
        if not (
            existing["strategy"] == result["strategy"]
            and existing["seed"] == result["seed"]
        )
    ]

    results.append(result)

    with result_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
        )


def run_single_experiment(
    strategy: str,
    seed: int,
    config: dict,
    tokenizer,
    train_dataset,
    eval_dataset,
    train_tokenized,
    eval_tokenized,
    embeddings,
    semantic_index,
) -> dict:
    print(
        f"\nRunning strategy={strategy}, seed={seed}"
    )

    # Reproduce current Group 1 seed behavior.
    set_experiment_seed(seed)

    model = build_lora_seq2seq_model(
        model_name=config["model"]["name"],
        r=config["lora"]["r"],
        alpha=config["lora"]["alpha"],
        dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
    )

    training_config = config["training"]

    # 300 optimizer updates × 2 accumulation steps
    # = 600 physical batches in the current 1K protocol.
    n_batches = (
        training_config["max_steps"]
        * training_config["gradient_accumulation_steps"]
    )

    order = build_batch_order(
        strategy=strategy,
        embeddings=embeddings,
        index=semantic_index,
        n_examples=len(train_dataset),
        batch_size=training_config[
            "per_device_train_batch_size"
        ],
        top_k=config["semantic_grouping"]["top_k"],
        n_batches=n_batches,
        seed=seed,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
    )

    experiment_name = config["experiment"]["name"]

    trainer_output_dir = (
        REPO_ROOT
        / "outputs"
        / "group1"
        / experiment_name
        / "trainer"
        / f"{strategy}_seed_{seed}"
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(trainer_output_dir),
        per_device_train_batch_size=training_config[
            "per_device_train_batch_size"
        ],
        gradient_accumulation_steps=training_config[
            "gradient_accumulation_steps"
        ],
        per_device_eval_batch_size=training_config[
            "per_device_eval_batch_size"
        ],
        max_steps=training_config["max_steps"],
        learning_rate=training_config["learning_rate"],
        logging_steps=training_config["logging_steps"],
        eval_strategy="no",
        save_strategy="no",
        seed=seed,
        report_to="none",
        predict_with_generate=True,
        fp16=training_config["fp16"],
    )

    trainer = OrderedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        processing_class=tokenizer,
        fixed_order=order,
    )

    trainer.train()

    eval_results = trainer.evaluate()

    result = {
        "strategy": strategy,
        "seed": seed,
        "eval_loss": eval_results.get("eval_loss"),
    }

    generation_metrics = evaluate_generation(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        eval_tokenized=eval_tokenized,
        max_target_length=config[
            "tokenization"
        ]["max_target_length"],
        generation_batch_size=64,
    )

    result.update(generation_metrics)

    print(result)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Group 1 batching experiments."
    )

    parser.add_argument(
        "--config",
        default="configs/group1/dolly_1k.yaml",
        help="Path to the Group 1 YAML configuration.",
    )

    parser.add_argument(
        "--strategy",
        default=None,
        help="Optionally run only one configured strategy.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optionally run only one configured seed.",
    )

    args = parser.parse_args()

    config_path = REPO_ROOT / args.config
    config = load_config(config_path)

    strategies = config[
        "experiment_design"
    ]["strategies"]

    seeds = config[
        "experiment_design"
    ]["seeds"]

    if args.strategy is not None:
        if args.strategy not in strategies:
            raise ValueError(
                f"Strategy {args.strategy!r} is not in the config."
            )
        strategies = [args.strategy]

    if args.seed is not None:
        if args.seed not in seeds:
            raise ValueError(
                f"Seed {args.seed} is not in the config."
            )
        seeds = [args.seed]

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"]
    )

    train_dataset, eval_dataset = load_dolly_split(
        dataset_name=config["dataset"]["name"],
        subset_size=config["dataset"]["subset_size"],
        shuffle_seed=config["dataset"]["shuffle_seed"],
        eval_ratio=config["dataset"]["eval_ratio"],
        split_seed=config["dataset"]["split_seed"],
    )

    train_tokenized, eval_tokenized = tokenize_dolly_split(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_input_length=config[
            "tokenization"
        ]["max_input_length"],
        max_target_length=config[
            "tokenization"
        ]["max_target_length"],
    )

    embeddings, semantic_index = build_semantic_index(
        train_dataset=train_dataset,
        embedding_model_name=config[
            "semantic_grouping"
        ]["embedding_model"],
        instruction_only=config[
            "semantic_grouping"
        ]["embed_instruction_only"],
    )

    result_path = (
        REPO_ROOT
        / "outputs"
        / "group1"
        / config["experiment"]["name"]
        / "per_seed_results.json"
    )

    for strategy in strategies:
        for seed in seeds:
            result = run_single_experiment(
                strategy=strategy,
                seed=seed,
                config=config,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                train_tokenized=train_tokenized,
                eval_tokenized=eval_tokenized,
                embeddings=embeddings,
                semantic_index=semantic_index,
            )

            save_result(
                result_path=result_path,
                result=result,
            )

    print(
        f"\nCompleted. Raw results saved to:\n{result_path}"
    )


if __name__ == "__main__":
    main()