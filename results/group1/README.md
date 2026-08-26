# Group 1 Results

Reviewed metrics from 36 matched semantic/random batching runs:

```text
3 dataset sizes × 4 strategies × 3 seeds = 36 runs
```

Strategies: `random`, `grouped`, `grouped_to_random`, `random_to_grouped`.

No single batching strategy is best across all dataset sizes and metrics.
Differences are generally small relative to seed-to-seed variation, so random
batching remains a strong baseline and semantic grouping does not show a
consistent performance advantage under this protocol.

Verified metrics: evaluation loss and ROUGE-1/2/L. BERTScore is not included.
No formal statistical-significance claim is made.
