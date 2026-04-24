# src/training/trainer_factory.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.batching.samplers import make_batch_sampler


@dataclass(frozen=True)
class TrainerComponents:
    model: torch.nn.Module
    tokenizer: Any
    trainer: Seq2SeqTrainer


class CustomBatchingSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        sampler_mode: str,
        train_batch_size: int,
        sampler_seed: int,
        anchor_to_group: dict[int, list[int]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._sampler_mode = sampler_mode
        self._train_batch_size = train_batch_size
        self._sampler_seed = sampler_seed
        self._anchor_to_group = anchor_to_group

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        batch_sampler = make_batch_sampler(
            sampler_mode=self._sampler_mode,
            dataset_size=len(self.train_dataset),
            batch_size=self._train_batch_size,
            seed=self._sampler_seed,
            drop_last=False,
            anchor_to_group=self._anchor_to_group,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )


def load_tokenizer(base_model_path: str | Path):
    return AutoTokenizer.from_pretrained(str(base_model_path))


def load_base_model(base_model_path: str | Path) -> torch.nn.Module:
    return AutoModelForSeq2SeqLM.from_pretrained(str(base_model_path))


def apply_lora(
    model: torch.nn.Module,
    lora_cfg: dict[str, Any],
) -> torch.nn.Module:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    return get_peft_model(model, peft_config)


def load_model_for_phase(
    *,
    base_model_path: str | Path,
    lora_cfg: dict[str, Any],
    init_adapter_path: str | Path | None = None,
) -> torch.nn.Module:
    """
    Phase initialization logic:
    - if init_adapter_path is None: start fresh from base model + new LoRA adapter
    - else: load base model and attach previously trained adapter
    """
    model = load_base_model(base_model_path)

    if init_adapter_path is None:
        model = apply_lora(model, lora_cfg)
    else:
        model = PeftModel.from_pretrained(model, str(init_adapter_path), is_trainable=True)

    return model


def make_training_arguments(
    output_dir: str | Path,
    training_cfg: dict[str, Any],
    learning_rate: float,
    num_train_epochs: float,
) -> Seq2SeqTrainingArguments:
    return Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        weight_decay=training_cfg.get("weight_decay", 0.0),
        logging_steps=training_cfg.get("logging_steps", 10),
        eval_strategy=training_cfg.get("eval_strategy", "steps"),
        eval_steps=training_cfg.get("eval_steps", 50),
        save_strategy=training_cfg.get("save_strategy", "epoch"),
        save_total_limit=training_cfg.get("save_total_limit", 2),
        predict_with_generate=training_cfg.get("predict_with_generate", False),
        bf16=training_cfg.get("bf16", False),
        fp16=training_cfg.get("fp16", False),
        report_to=training_cfg.get("report_to", []),
        remove_unused_columns=False,
    )


def build_trainer(
    *,
    base_model_path: str | Path,
    output_dir: str | Path,
    train_dataset,
    eval_dataset,
    lora_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    sampler_mode: str,
    sampler_seed: int,
    learning_rate: float,
    num_train_epochs: float,
    anchor_to_group: dict[int, list[int]] | None = None,
    init_adapter_path: str | Path | None = None,
) -> TrainerComponents:
    tokenizer = load_tokenizer(base_model_path)
    model = load_model_for_phase(
        base_model_path=base_model_path,
        lora_cfg=lora_cfg,
        init_adapter_path=init_adapter_path,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    args = make_training_arguments(
        output_dir=output_dir,
        training_cfg=training_cfg,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
    )

    trainer = CustomBatchingSeq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        sampler_mode=sampler_mode,
        train_batch_size=training_cfg["per_device_train_batch_size"],
        sampler_seed=sampler_seed,
        anchor_to_group=anchor_to_group,
    )

    return TrainerComponents(
        model=model,
        tokenizer=tokenizer,
        trainer=trainer,
    )