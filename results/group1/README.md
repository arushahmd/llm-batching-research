# Group 1 Results

This directory contains lightweight, reviewed metrics extracted from the
executed Group 1 notebooks.

No experiment was rerun to create these files. The values were consolidated
from the preserved 1K, 3K, and 5K multi-seed notebooks.

## Files

### `per_seed_results.csv`

Contains one row for every completed Group 1 run.

The standardized experiment family contains:

```text
3 dataset sizes
× 4 batching strategies
× 3 random seeds
= 36 runs
```

Columns:

- `dataset_label`
- `dataset_size`
- `strategy`
- `seed`
- `eval_loss`
- `rouge1`
- `rouge2`
- `rougeL`

### `summary.csv`

Aggregates the three seeds for every dataset-size / strategy combination.

For each metric it reports:

- arithmetic mean
- sample standard deviation across the three seeds

This produces 12 aggregate rows:

```text
3 dataset sizes × 4 strategies = 12 summaries
```

## Dataset Sizes

The standardized Group 1 conditions are:

- Dolly 1K
- Dolly 3K
- Dolly 5K

All three use the canonical thesis-v2 settings documented in:

```text
configs/group1/
docs/experiment-protocol.md
```

## Strategies

The four Group 1 strategies are:

- `random`
- `grouped`
- `grouped_to_random`
- `random_to_grouped`

## Metrics

The verified metrics included here are:

- evaluation loss
- ROUGE-1 F1
- ROUGE-2 F1
- ROUGE-L F1

BERTScore is intentionally not included because it was not calculated by the
verified Group 1 evaluation implementation.

## Interpretation

These files are descriptive research artifacts.

Final claims should be based on the multi-seed aggregate behavior rather than
on a single best run. Differences should also be interpreted in the context of
seed-to-seed variation.

The contrastive-learning extension is a separate experimental block and should
not overwrite or modify these Group 1 baseline results.
