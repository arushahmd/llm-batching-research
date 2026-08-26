# Group 2 — Length-Curriculum Results

Reviewed metrics from 18 matched runs:

```text
3 dataset sizes × 2 curriculum directions × 3 seeds = 18 runs
```

Directions: `easy_to_hard` and `hard_to_easy`.

The difficulty proxy is the **full tokenized instruction+context input length
before training-time truncation**. Examples are fully sorted, and the order is
cycled when necessary to fill the fixed training sequence.

## Finding

Hard → Easy has lower **mean evaluation loss** at 1K, 3K, and 5K.

ROUGE is scale-dependent:

- 1K: Easy → Hard is higher on mean ROUGE-1/2/L.
- 3K: Easy → Hard is higher on mean ROUGE-1/2/L.
- 5K: Hard → Easy is higher on mean ROUGE-1/2/L.

The 5K Hard → Easy generation scores also show larger seed variability.
Accordingly, no universal curriculum-direction superiority is claimed.

No formal statistical-significance claim is made.

## Files

- `per_seed_results.csv`
- `summary.csv`
- `figures/`

## Notebook Provenance

The public Group 2 notebooks are curated source-only copies of the executed
Drive notebooks. Outputs are stripped. The raw 1K Drive notebook contains a
stale 3K data-loading output from a later re-execution/edit; verified metrics
are preserved in this directory and the canonical Drive tracker.
