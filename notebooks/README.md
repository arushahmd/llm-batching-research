# Notebooks

```text
00_validation/   pipeline validation
01_group1/       semantic/random batching experiments
02_group2/       length-curriculum experiments
```

Stable reusable logic lives under `src/`.

The Group 2 public notebooks are **source-only curated copies**: outputs and
execution counts are stripped. The raw 1K Drive artifact contains a stale 3K
data-loading output produced by a later re-execution/edit; the source
configuration is 1K and verified metrics are maintained under `results/group2/`.

Large runtime artifacts, notebook checkpoints, model checkpoints, caches, and
prediction dumps are not committed.
