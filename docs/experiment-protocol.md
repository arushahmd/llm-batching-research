# Experiment Protocol

## Shared Matched Setup

- Model: `google/flan-t5-small`
- Dataset: `databricks/databricks-dolly-15k`
- Subsets: 1K, 3K, 5K
- Shuffle seed: 42
- Evaluation ratio: 0.10
- Split seed: 42
- Max input length: 256
- Max target length: 128
- LoRA: r=8, alpha=16, dropout=0.05, q/v targets
- Physical batch size: 3
- Gradient accumulation: 2
- Evaluation batch size: 2
- Max optimizer steps: 300
- Learning rate: 3e-3
- fp16: false
- Seeds: 13, 21, 42
- Metrics: eval loss, ROUGE-1/2/L
- Generation batch size: 64
- Max generated tokens: 128

Each run constructs 600 physical batches and therefore 1,800 fixed-order
example positions.

### Target-Padding Reproducibility

Completed experiments keep padded target token IDs in the label sequence rather
than replacing them with `-100`. The curated implementation preserves this
executed behavior.

## Group 1 — Semantic / Random Batching

Configs: `configs/group1/`.

Strategies:

- random
- grouped
- grouped → random
- random → grouped

Semantic grouping uses `all-MiniLM-L6-v2`, L2-normalized embeddings,
`faiss.IndexFlatIP`, instruction+context text, and `top_k=8`.

The two-phase schedules split the 600 physical batches into 300/300 and use
`seed + 1000` for the second phase.

```text
3 sizes × 4 strategies × 3 seeds = 36 runs
```

## Group 2 — Length Curriculum

Configs: `configs/group2/`.

Directions:

- easy → hard
- hard → easy

Difficulty proxy:

```python
len(tokenizer(input_text)["input_ids"])
```

This is the **full tokenizer output length before the training-time 256-token
truncation**.

The index list is fully sorted with no within-order shuffling and is cycled if
needed to fill all 1,800 positions.

```text
3 sizes × 2 directions × 3 seeds = 18 runs
```

## Reviewed Experiment Set

```text
54 runs total
```

Historical thesis-v1 experiments used materially different settings and remain
isolated from this matched protocol.
