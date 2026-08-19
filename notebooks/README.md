# Notebooks

This directory contains the research notebooks used for pipeline validation,
experiment execution, and analysis.

The notebooks are retained as research artifacts, while stable and reusable
implementation logic is maintained under `src/`.

## Structure

```text
notebooks/
├── 00_validation/
│   ├── alpaca_pipeline_validation.ipynb
│   └── dolly_pipeline_validation.ipynb
│
├── 01_group1/
│   ├── group1_dolly_1k_multiseed.ipynb
│   ├── group1_dolly_3k_multiseed.ipynb
│   └── group1_dolly_5k_multiseed.ipynb
│
└── README.md
```

## `00_validation/`

These notebooks were used to validate the research pipeline and inspect the
data-preparation and fine-tuning workflow before the main Group 1 experiments.

They should be treated as validation artifacts rather than the canonical
implementation of the current experiment pipeline.

## `01_group1/`

These notebooks contain the current Group 1 multi-seed experiments for the
Dolly 1K, 3K, and 5K subsets.

The Group 1 study compares four batching strategies:

- random
- grouped
- grouped-to-random
- random-to-grouped

with experiment seeds:

- 13
- 21
- 42

The current experimental protocol is documented through the YAML configurations
under `configs/group1/` and reusable implementation modules under `src/`.

## Notebook and Source-Code Roles

The project intentionally separates notebooks from reusable Python modules.

Notebooks are used for:

- experiment execution
- interactive inspection
- exploratory analysis
- validation
- visual review of intermediate outputs

Reusable Python modules are used for:

- dataset loading and tokenization
- semantic embedding and FAISS indexing
- batching strategy construction
- fixed-order training
- LoRA model initialization
- reproducibility utilities
- generation evaluation

This separation reduces duplicated logic across notebooks and makes the
experimental protocol easier to review and maintain.

## Current Status

The notebooks are preserved as research records while the revised thesis
protocol is being organized into a cleaner reproducible codebase.

The current 1K notebook is the primary reference implementation for Group 1.
The 3K and 5K notebooks remain part of the experimental record and will be
reviewed against the same canonical protocol before final results are published.

Historical notebooks from the earlier thesis protocol are not kept in the
active thesis-v2 notebook tree. They remain preserved through the repository's
legacy branch and tag.

## Important Reproducibility Note

The current completed Group 1 notebooks tokenize target sequences with
fixed-length padding as originally executed. The active thesis-v2 code preserves
that behavior for reproducibility.

Any change to target-label masking, training semantics, or other methodology
should be treated as a new protocol requiring an explicit rerun rather than as
a silent cleanup of existing experiments.

## Artifact Policy

Notebook checkpoints, temporary copies, raw exports, model checkpoints,
embedding caches, generated prediction dumps, and other large runtime artifacts
should not be committed to this directory.

Only notebooks that are useful for understanding, validating, or reproducing
the research should remain in the public repository.
