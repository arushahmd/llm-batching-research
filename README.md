# Semantic Batching for Instruction Fine-Tuning

An empirical research project studying whether the semantic composition and
ordering of mini-batches affects instruction fine-tuning.

The project compares standard random batching with embedding-based semantic
grouping and two curriculum-style training orders across controlled Dolly
subsets and multiple random seeds.

## Research Question

> Does organizing instruction-tuning examples into semantically coherent
> mini-batches improve optimization or generation quality compared with
> standard random batching?

The current Group 1 study evaluates four strategies:

- **Random**
- **Semantically Grouped**
- **Grouped → Random**
- **Random → Grouped**

## Experimental Design

### Model

- `google/flan-t5-small`
- LoRA parameter-efficient fine-tuning

### Dataset

- `databricks/databricks-dolly-15k`
- Controlled subsets:
  - 1K
  - 3K
  - 5K

### Semantic Retrieval

- Sentence embeddings: `all-MiniLM-L6-v2`
- Retrieval: FAISS
- Normalized inner-product search for cosine-similarity ranking
- Instruction + context embeddings
- Semantic neighbor pool: `top_k = 8`

### Training

- Seeds: `13`, `21`, `42`
- Physical batch size: `3`
- Gradient accumulation: `2`
- Evaluation batch size: `2`
- Maximum optimizer steps: `300`
- Learning rate: `3e-3`
- LoRA rank: `8`
- LoRA alpha: `16`
- LoRA dropout: `0.05`
- LoRA targets: `q`, `v`

For each dataset size:

```text
4 strategies × 3 seeds = 12 runs
```

Across all three standardized dataset sizes:

```text
12 runs × 3 dataset sizes = 36 Group 1 runs
```

## Evaluation

The verified Group 1 pipeline records:

- Evaluation loss
- ROUGE-1
- ROUGE-2
- ROUGE-L

> BERTScore is not part of the verified Group 1 evaluation pipeline.

## Repository Structure

```text
.
├── configs/
│   └── group1/
│       ├── dolly_1k.yaml
│       ├── dolly_3k.yaml
│       └── dolly_5k.yaml
│
├── docs/
│   ├── README.md
│   ├── research-question.md
│   └── experiment-protocol.md
│
├── notebooks/
│   ├── 00_validation/
│   └── 01_group1/
│
├── results/
│   └── reviewed lightweight result artifacts
│
├── scripts/
│   └── run_group1.py
│
├── src/
│   ├── batching/
│   ├── data/
│   ├── evaluation/
│   ├── training/
│   └── utils/
│
├── tests/
│   └── batching strategy tests
│
└── archive/
    └── historical or superseded material
```

## Repository Design

The repository intentionally separates working research artifacts from reusable
implementation.

### `notebooks/`

Research notebooks are retained for:

- experiment execution
- validation
- exploratory analysis
- intermediate inspection
- preservation of the executed research record

### `src/`

Reusable implementation is extracted into Python modules for:

- Dolly dataset preparation
- semantic text construction
- SentenceTransformer embeddings
- FAISS indexing
- random batching
- semantic grouped batching
- two-phase curriculum ordering
- fixed-order training
- LoRA model initialization
- reproducibility utilities
- generation evaluation

### `configs/`

The 1K, 3K, and 5K Group 1 conditions are represented as explicit YAML
configurations.

The three configurations intentionally share the same core experimental
protocol, with dataset subset size as the controlled difference.

### `docs/`

Human-readable research design is documented separately from implementation
code.

See:

- `docs/research-question.md`
- `docs/experiment-protocol.md`

### `results/`

This directory is reserved for lightweight, reviewed artifacts such as:

- per-seed metrics
- aggregate summaries
- selected tables
- selected figures

Raw checkpoints, caches, embeddings, and temporary experiment outputs are kept
outside version control.

## Current Research Status

The original research repository has been reorganized around a standardized
thesis-v2 protocol.

The current Group 1 experiment family consists of 36 runs:

- 4 batching strategies
- 3 random seeds
- 3 Dolly subset sizes

The 1K, 3K, and 5K conditions use the same core model, training, LoRA,
semantic-retrieval, and evaluation settings.

The executed notebooks are preserved under `notebooks/01_group1/`. Their metrics
are being consolidated into lightweight reviewed artifacts under `results/`.

Historical experiments that used materially different settings remain isolated
from the active thesis-v2 protocol.

## Reproducibility Principle

The repository follows a strict distinction between preserving completed
experiments and improving methodology:

```text
completed protocol → preserve exactly for reproducibility
methodology change → document separately and rerun
```

One important example is target padding.

The completed Group 1 protocol keeps padded target token IDs in the label
sequence rather than replacing padding positions with `-100`. The curated code
preserves that executed behavior.

Changing this would define a different training protocol and therefore requires
an explicit rerun rather than a silent code cleanup.

## Contrastive-Learning Extension

The next planned research block investigates a stronger contrastive-learning
formulation.

Group 1 changes the composition and order of examples while retaining the
standard supervised sequence-to-sequence objective.

The contrastive extension will be treated as a separate experiment because it
introduces additional learning structure, such as explicit positive and
negative relationships and a contrastive loss.

This separation keeps the Group 1 baseline scientifically interpretable.

## Research Workflow

```text
raw / working research artifacts
            ↓
inspection and validation
            ↓
reusable implementation
            ↓
reviewed result artifacts
            ↓
public research repository
```

Large and transient artifacts are intentionally excluded, including:

- model checkpoints
- LoRA adapter checkpoints
- raw datasets
- Hugging Face caches
- embedding caches
- FAISS index files
- temporary experiment outputs
- generated prediction dumps
- runtime logs

## Versioning

The active thesis-v2 work is maintained separately from the earlier research
protocol.

Earlier experiments used materially different settings and should not be mixed
with the standardized Group 1 results unless explicitly identified as legacy
comparisons.

## Author

**Aroosh Ahmad**

MPhil Artificial Intelligence

Research interests: Large Language Models, NLP, instruction fine-tuning,
retrieval, contrastive learning, and applied AI systems.
