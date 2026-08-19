from __future__ import annotations

import torch
from rouge_score import rouge_scorer


def evaluate_generation(
    model,
    tokenizer,
    eval_dataset,
    eval_tokenized,
    max_target_length: int,
    generation_batch_size: int = 64,
) -> dict[str, float]:
    """
    Generate responses for the evaluation split and compute
    ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    This reproduces the current Group 1 notebook evaluation protocol.
    """
    model.eval()

    predictions: list[str] = []

    references = [
        eval_dataset[i]["target_text"]
        for i in range(len(eval_dataset))
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        for start in range(
            0,
            len(eval_dataset),
            generation_batch_size,
        ):
            end = min(
                start + generation_batch_size,
                len(eval_dataset),
            )

            batch_input_ids = torch.tensor(
                [
                    eval_tokenized[i]["input_ids"]
                    for i in range(start, end)
                ]
            ).to(device)

            batch_attention_mask = torch.tensor(
                [
                    eval_tokenized[i]["attention_mask"]
                    for i in range(start, end)
                ]
            ).to(device)

            generated = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=max_target_length,
            )

            batch_predictions = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )

            predictions.extend(batch_predictions)

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for prediction, reference in zip(
        predictions,
        references,
    ):
        scores = scorer.score(
            reference,
            prediction,
        )

        rouge1_scores.append(
            scores["rouge1"].fmeasure
        )
        rouge2_scores.append(
            scores["rouge2"].fmeasure
        )
        rougeL_scores.append(
            scores["rougeL"].fmeasure
        )

    return {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores),
        "rouge2": sum(rouge2_scores) / len(rouge2_scores),
        "rougeL": sum(rougeL_scores) / len(rougeL_scores),
    }