# Semantic Batching for Instruction Fine-Tuning

Research repository for studying how semantic mini-batch construction and
curriculum-style batch ordering affect instruction fine-tuning.

The project compares standard random batching with embedding-based semantic
grouping and two-phase batching curricula under a controlled multi-seed setup.

## Research Question

Does organizing instruction-tuning examples into semantically coherent
mini-batches improve optimization or generation quality compared with
standard random batching?

The current study evaluates four batching strategies:

- Random
- Semantically Grouped
- Grouped → Random
- Random → Grouped

## Current Experimental Setting

### Model

- `google/flan-t5-small`
- Parameter-efficient fine-tuning with LoRA

### Dataset

- `databricks/databricks-dolly-15k`
- Controlled subsets:
  - 1K
  - 3K
  - 5K

### Semantic Grouping

- Sentence embeddings: `all-MiniLM-L6-v2`
- Similarity search: FAISS
- Cosine similarity through normalized embeddings and inner-product search
- Anchor-based semantic neighbor sampling

### Current Group 1 Training Protocol

- Seeds: `13`, `21`, `42`
- Physical training batch size: `3`
- Gradient accumulation steps: `2`
- Evaluation batch size: `2`
- Maximum optimizer steps: `300`
- Learning rate: `3e-3`
- LoRA rank: `8`
- LoRA alpha: `16`
- LoRA dropout: `0.05`
- LoRA target modules: `q`, `v`
- Semantic neighbor pool: `top_k = 8`

Experiment parameters are stored explicitly in YAML configuration files.

## Evaluation

The current Group 1 protocol records:

- Evaluation loss
- ROUGE-1
- ROUGE-2
- ROUGE-L

Generation metrics are computed from model-generated responses against the
held-out reference responses.

> BERTScore is not currently part of the verified Group 1 evaluation pipeline.

## Repository Structure

```text
.
├── configs/
│   └── group1/
│       └── dolly_1k.yaml
│
├── docs/
│   └── research documentation
│
├── notebooks/
│   ├── 00_validation/
│   └── 01_group1/
│
├── results/
│   └── curated and validated result artifacts
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

## Code Organization

The repository separates experimental notebooks from reusable research code.

### `notebooks/`

Used for:

- exploratory research
- validation
- experiment execution
- inspection of intermediate outputs
- research analysis

### `src/`

Contains reusable and curated implementations for:

- Dolly dataset preparation
- semantic embedding and FAISS indexing
- batching strategies
- deterministic experiment ordering
- LoRA model initialization
- generation evaluation
- reproducibility utilities

### `configs/`

Stores experiment definitions separately from implementation code so that
dataset size, model settings, training parameters, and strategy choices remain
explicit and auditable.

### `results/`

Reserved for lightweight, reviewed research outputs such as:

- per-seed metrics
- aggregate summaries
- selected tables
- selected figures

Raw checkpoints, model weights, caches, embeddings, and temporary experiment
artifacts are not intended to be committed to the repository.

## Research Workflow

The repository follows a separation between working research artifacts and
curated source-controlled artifacts:

```text
Working notebooks / raw research artifacts
                ↓
        inspection and validation
                ↓
       reusable implementation
                ↓
        curated GitHub repository
```

This allows notebooks to remain useful for research while stable logic is
maintained as reusable Python modules.

## Current Status

The repository is currently being reorganized around the revised experimental
protocol.

Completed repository work includes:

- preservation of the previous research version
- canonical Group 1 configuration
- modular Dolly data pipeline
- semantic embedding and FAISS utilities
- random and semantic batching implementations
- two-phase batching curriculum implementations
- fixed-order Hugging Face trainer
- LoRA model construction
- experiment seed utilities
- generation and ROUGE evaluation
- batching strategy tests
- configurable Group 1 experiment runner

The 1K, 3K, and 5K experimental artifacts are being reviewed and organized
before final aggregate results and research conclusions are published.

## Results Status

Final thesis-v2 conclusions are intentionally not stated yet.

Historical experiments and current reruns use materially different protocols,
so results from earlier work should not be mixed with the current experimental
design.

Final claims will be added only after the complete current experiment set has
been validated and aggregated.

## Reproducibility

The revised repository is designed around:

- explicit YAML experiment configurations
- fixed dataset split seeds
- explicit experiment seeds
- fresh model initialization per run
- deterministic construction of batch orders
- reusable implementation modules
- separation of raw and curated results

Exact environment packaging and full end-to-end reproduction instructions will
be finalized after the experimental environment is validated.

## Artifact Policy

Large or transient research artifacts should remain outside GitHub, including:

- model checkpoints
- LoRA adapter checkpoints
- Hugging Face caches
- raw datasets
- embedding caches
- FAISS index files
- temporary experiment outputs
- runtime logs
- generated prediction dumps

GitHub is reserved for code, configurations, selected notebooks, documentation,
tests, and validated lightweight research results.

## Author

**Aroosh Ahmad**

MPhil Artificial Intelligence

Research interests: Large Language Models, NLP, instruction fine-tuning,
retrieval, and applied AI systems.
