# Research Documentation

This directory contains concise documentation for the current thesis-v2
experimental protocol.

The purpose of these documents is to keep the research design, methodology,
and experimental assumptions separate from implementation code and notebooks.

## Planned Documentation

### `research-question.md`

Defines:

- the primary research question
- the motivation for semantic batching
- the comparison strategies
- the scope of the current study
- the claims the experiments are designed to test

### `experiment-protocol.md`

Documents the canonical experimental setup, including:

- model
- dataset and subset sizes
- train/evaluation split
- tokenization lengths
- LoRA configuration
- batching strategies
- random seeds
- training hyperparameters
- evaluation metrics

This file should be treated as the human-readable protocol corresponding to
the YAML experiment configurations under `configs/`.

### `methodology.md`

Explains how the experiment is implemented, including:

- Dolly example formatting
- semantic embedding construction
- FAISS nearest-neighbor retrieval
- random batch-order generation
- semantic grouped batch-order generation
- two-phase curriculum ordering
- fixed-order training
- generation evaluation

## Source of Truth

Different repository components serve different purposes:

```text
configs/       machine-readable experiment parameters
src/           reusable implementation
scripts/       experiment orchestration
notebooks/     research execution and analysis artifacts
docs/          human-readable research design
results/       reviewed lightweight results
```

The YAML configuration and current implementation should remain consistent with
the documented protocol.

If a methodology change would alter the behavior of already completed
experiments, it should be treated as a new protocol or explicit rerun rather
than silently modifying the historical experiment definition.

## Versioning Note

The active `restructure/thesis-v2` branch represents the revised research
organization.

Earlier thesis experiments used materially different configurations and are
preserved separately through the repository's legacy branch and tag. Historical
results should not be mixed with thesis-v2 results unless they are explicitly
identified as legacy comparisons.

## Current Status

The documentation is being written alongside repository cleanup.

Final experimental conclusions are intentionally excluded until the complete
current protocol and result set have been reviewed and validated.
