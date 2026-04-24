# Instruction Fine-Tuning via Semantic Batching

## Overview

This repository presents an empirical study on whether **semantic batching** improves instruction fine-tuning of large language models compared to standard **random batching**.

The core idea is to construct mini-batches using semantically similar instruction samples, retrieved through embedding-based nearest-neighbor search, with the goal of producing more coherent gradient updates during training.

> Instead of sampling training examples randomly, semantic batching groups related instructions together and evaluates whether this improves convergence, generalization, or generation quality.

---

## Research Objective

The objective of this project is to evaluate whether **retrieval-based semantic mini-batch construction** improves instruction fine-tuning outcomes compared to random batching.

Specifically, this work investigates whether semantic batching:

- improves convergence behavior
- reduces evaluation loss
- improves generalization
- reduces training variance
- improves generation quality
- interacts meaningfully with curriculum-style two-phase training

---

## Methodology

### Model

- Base model: `google/flan-t5-small`
- Fine-tuning method: LoRA
- Training framework: Hugging Face Transformers

### Dataset

The experiments use subsets of the Dolly instruction dataset:

- Dolly 1k
- Dolly 3k
- Dolly 5k

### Semantic Grouping

Semantic grouping is performed using embedding-based retrieval:

- Embedding model: `all-MiniLM-L6-v2`
- Similarity search: FAISS
- Grouping strategy: anchor sample + nearest semantic neighbors

### Training Setup

The experiments use a controlled multi-seed setup:

- Seeds: `13`, `21`, `42`
- Two-phase training design:
  - Phase 1
  - Phase 2

### Batching Strategies

The following batching strategies are compared:

- Random
- Grouped
- Random → Grouped
- Grouped → Random

### Evaluation Metrics

The project evaluates both optimization and generation behavior.

Optimization metrics:

- Evaluation loss
- Training loss
- Phase-wise evaluation improvement
- Generalization gap

Generation metrics:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- BERTScore

---

## Experimental Blocks

The experiments are organized into progressively stronger evaluation blocks:

| Block | Dataset | Training Setup | Purpose |
|---|---:|---|---|
| 1k Short | Dolly 1k | Short two-phase training | Initial controlled signal check |
| 1k Long | Dolly 1k | Longer two-phase training | Test whether early signal persists |
| 3k Long | Dolly 3k | Longer two-phase training | Scaling validation |
| 5k Long | Dolly 5k | Longer two-phase training | Strongest validation setting |

---

## Key Findings

### 1. Random batching is a strong baseline

Random batching consistently matched or outperformed structured semantic batching strategies across the controlled experiments.

### 2. Semantic grouping does not improve generalization

No consistent reduction in evaluation loss was observed across dataset sizes, training durations, or random seeds.

### 3. Curriculum-style batching provides no stable benefit

Switching batching strategies across phases, such as Random → Grouped or Grouped → Random, did not produce a reliable improvement.

### 4. The batching effect diminishes with scale

The early weak signal observed in smaller experiments becomes negligible as dataset size increases.

### 5. Generation quality does not improve meaningfully

Generation evaluation using ROUGE and BERTScore did not show a consistent advantage for semantic grouping or curriculum batching.

---

## Final Conclusion

Semantic batching does not provide a meaningful advantage over random batching for instruction fine-tuning in this experimental setting.

Random batching remains:

- simple
- robust
- effective
- difficult to outperform

This project is therefore an empirical negative result: a theoretically plausible training strategy was tested systematically and found not to improve optimization, generalization, or generation quality under the evaluated conditions.

---

## Repository Structure

```text
configs/        # Experiment and model configuration files
src/            # Core training, batching, data, and evaluation logic
scripts/        # Runnable scripts for data processing, training, evaluation, and reporting
manifests/      # Project path and asset manifests
metadata/       # Experiment registry and tracking metadata
notebooks/      # Exploratory and pipeline notebooks

reports/
  ├── master/                      # Official aggregated master results
  ├── plots/                       # Final result visualizations
  ├── generation_eval_summaries/   # Generation evaluation summaries
  └── archive/                     # Older block-wise summaries
```
## Running Experiments

### Run a multi-seed experiment

```
python-m scripts.training.run_multiseed \
--config configs/experiments/exp_026_random_only_multiseed_5k.yaml
```

### Aggregate results

```
python-m scripts.reporting.aggregate_results
```

### Generate plots

```
python-m scripts.reporting.generate_plots
```

### Evaluate generation quality

```
python-m scripts.evaluation.evaluate_generation_quality
```

---

## Data and Models

Due to size constraints, datasets, trained models, checkpoints, semantic indexes, and full experiment outputs are stored externally.

Google Drive:

```
[LINK WILL BE PROVIDED IN FINAL REPOSITORY]
```

External artifact structure:

```
data/
models/
experiments/
reports/
exports/
```

The GitHub repository keeps only lightweight, reproducible assets such as code, configurations, notebooks, summary tables, and plots.

---

## Reproducibility

This repository is structured to support reproducible experimentation through:

- fixed random seeds
- YAML-based experiment configurations
- modular training and batching components
- explicit project manifests
- separated code and artifact storage
- aggregated result tables
- preserved experiment registry

---

## Future Work

Potential extensions include:

- difficulty-aware curriculum batching
- task-aware instruction grouping
- loss-aware sampling strategies
- hybrid semantic + difficulty-based curricula
- evaluation on larger models such as T5-base or LLaMA-family models
- evaluation on harder or more diverse instruction datasets
- deeper qualitative analysis of generation behavior

---

## Research Significance

This project contributes a systematic empirical evaluation of semantic batching for instruction fine-tuning.

Although semantic batching was expected to improve training dynamics, the results show that random batching remains a strong and reliable baseline. This negative result is useful because it helps clarify the limits of intuitive batching strategies and prevents overclaiming benefits from semantic grouping without rigorous validation.

---

## Author

**Aroosh Ahmad**

MPhil Artificial Intelligence

Focus: NLP, LLMs, Instruction Fine-Tuning, and Agentic AI Systems