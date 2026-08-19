from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM


def build_lora_seq2seq_model(
    model_name: str,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
):
    """
    Load a fresh sequence-to-sequence model and attach LoRA adapters.

    A new base model is loaded on every call so that each experiment run
    starts from an independent fresh model state.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
    )

    return get_peft_model(model, lora_config)