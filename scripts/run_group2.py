from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.batching.length_curriculum import (  # noqa: E402
    build_length_order,
    compute_input_lengths,
)
from src.data.dolly import (  # noqa: E402
    load_dolly_split,
    tokenize_dolly_split,
)
from src.evaluation.generation import evaluate_generation  # noqa: E402
from src.training.model import build_lora_seq2seq_model  # noqa: E402
from src.training.ordered_trainer import OrderedTrainer  # noqa: E402
from src.utils.reproducibility import set_experiment_seed  # noqa: E402


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_executed_protocol(config: dict) -> None:
    """Validate config fields whose alternatives are not part of Group 2."""
    tokenization_config = config["tokenization"]

    if tokenization_config["mask_target_padding"] is not False:
        raise ValueError(
            "Group 2 preserves padded target token IDs; "
            "mask_target_padding must be false."
        )

    length_config = config["length_curriculum"]
    expected_settings = {
        "difficulty_proxy": "input_token_length",
        "text_source": "instruction_and_context",
        "apply_training_max_length_to_difficulty": False,
        "ordering": "fully_sorted",
        "cycle_if_needed": True,
    }

    for key, expected in expected_settings.items():
        actual = length_config[key]

        if actual != expected:
            raise ValueError(
                f"Unsupported Group 2 setting {key}={actual!r}; "
                f"the executed protocol requires {expected!r}."
            )


def save_result(
    result_path: Path,
    result: dict,
) -> None:
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
            existing["method"] == result["method"]
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
    method: str,
    seed: int,
    config: dict,
    tokenizer,
    train_dataset,
    eval_dataset,
    train_tokenized,
    eval_tokenized,
    lengths: list[int],
) -> dict:
    set_experiment_seed(seed)

    model = build_lora_seq2seq_model(
        model_name=config["model"]["name"],
        r=config["lora"]["r"],
        alpha=config["lora"]["alpha"],
        dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
    )

    training_config = config["training"]
    n_batches = (
        training_config["max_steps"]
        * training_config["gradient_accumulation_steps"]
    )
    order = build_length_order(
        lengths=lengths,
        direction=method,
        n_batches=n_batches,
        batch_size=training_config["per_device_train_batch_size"],
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
    )
    trainer_output_dir = (
        REPO_ROOT
        / "outputs"
        / "group2"
        / config["experiment"]["name"]
        / "trainer"
        / f"{method}_seed_{seed}"
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
    result = {
        "method": method,
        "seed": seed,
        "eval_loss": trainer.evaluate().get("eval_loss"),
    }
    result.update(
        evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            eval_tokenized=eval_tokenized,
            max_target_length=config[
                "tokenization"
            ]["max_target_length"],
            generation_batch_size=config[
                "evaluation"
            ]["generation_batch_size"],
        )
    )

    del trainer, model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Group 2 length-curriculum experiments."
    )
    parser.add_argument(
        "--config",
        default="configs/group2/dolly_1k.yaml",
        help="Path to the Group 2 YAML configuration.",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Optionally run only one configured curriculum direction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optionally run only one configured seed.",
    )
    args = parser.parse_args()

    config = load_config(REPO_ROOT / args.config)
    validate_executed_protocol(config)
    methods = list(config["experiment_design"]["methods"])
    seeds = list(config["experiment_design"]["seeds"])

    if args.method is not None:
        if args.method not in methods:
            raise ValueError(
                f"Unknown configured method: {args.method}"
            )

        methods = [args.method]

    if args.seed is not None:
        if args.seed not in seeds:
            raise ValueError(
                f"Unknown configured seed: {args.seed}"
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
    lengths = compute_input_lengths(
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        instruction_only=False,
    )
    result_path = (
        REPO_ROOT
        / "outputs"
        / "group2"
        / config["experiment"]["name"]
        / "per_seed_results.json"
    )

    for method in methods:
        for seed in seeds:
            result = run_single_experiment(
                method=method,
                seed=seed,
                config=config,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                train_tokenized=train_tokenized,
                eval_tokenized=eval_tokenized,
                lengths=lengths,
            )
            print(result)
            save_result(
                result_path=result_path,
                result=result,
            )


if __name__ == "__main__":
    main()
