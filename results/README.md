# Results

This directory contains reviewed, lightweight research outputs suitable for
version control.

## Group 1 — Semantic Batching

[`group1/`](group1/) contains the standardized thesis-v2 semantic-batching
experiment family:

```text
3 Dolly subset sizes × 4 batching strategies × 3 seeds = 36 runs
```

Available artifacts:

- `group1/per_seed_results.csv` — one row per completed run
- `group1/summary.csv` — mean and sample standard deviation across three seeds
- `group1/figures/` — selected multi-seed visualizations
- `group1/README.md` — result interpretation and artifact notes

Only reviewed metrics and selected figures belong in this directory.

Model checkpoints, caches, embeddings, raw generated predictions, and temporary
training outputs should remain outside Git.
