# Research Question

## Primary Research Question

Does semantically coherent mini-batch construction improve instruction
fine-tuning performance compared with standard random batching under a
controlled multi-seed training setup?

The study focuses on whether the order and semantic composition of training
examples influence optimization behavior and downstream generation quality.

## Motivation

Standard mini-batch training usually relies on random sampling. Random batching
is simple, widely used, and provides a strong baseline, but it does not consider
the semantic relationship between examples within a batch.

This project investigates whether grouping semantically related instruction
examples can produce more coherent gradient updates during instruction
fine-tuning.

The underlying idea is:

```text
semantically related examples
            ↓
more coherent mini-batches
            ↓
different optimization dynamics
            ↓
potential change in generalization or generation quality
```

This is treated as an empirical question rather than an assumed advantage.

## Experimental Comparison

The current Group 1 study compares four training-order strategies.

### 1. Random

Each physical mini-batch is sampled randomly from the training set.

This serves as the primary baseline.

### 2. Grouped

Each batch begins with an anchor example.

Semantically similar examples are retrieved from the training set using
sentence embeddings and FAISS nearest-neighbor search. A subset of these
neighbors is sampled to complete the mini-batch.

### 3. Grouped → Random

The first half of the physical training batches use semantic grouping.

The second half use random batching.

This tests whether beginning with semantically coherent batches and later
introducing more diverse random batches affects training behavior.

### 4. Random → Grouped

The first half of the physical training batches use random batching.

The second half use semantic grouping.

This tests whether beginning with broad random exposure and later moving toward
more semantically coherent batches affects training behavior.

## Outcomes of Interest

The experiments are designed to evaluate whether batching strategy affects:

- final evaluation loss
- ROUGE-1
- ROUGE-2
- ROUGE-L
- consistency of results across random seeds

The current verified Group 1 pipeline does not include BERTScore.

## Working Hypotheses

The research is intentionally structured so that either positive or negative
results are meaningful.

### H1 — Semantic Grouping Hypothesis

Semantically grouped mini-batches may alter optimization behavior relative to
random batching because examples within the same physical batch are more
semantically coherent.

### H2 — Curriculum-Order Hypothesis

The order in which random and grouped batching are applied may influence the
final model because the optimization trajectory depends on the sequence of
training examples.

### H0 — Null Hypothesis

Semantic grouping and curriculum-style ordering may provide no consistent
advantage over random batching under the evaluated experimental setting.

The repository does not assume that H1 or H2 is correct.

## Controlled Variables

The batching strategy is intended to be the principal experimental variable.

Within a given dataset-size comparison, the following settings are controlled:

- base model
- dataset source
- subset construction
- train/evaluation split
- tokenization limits
- LoRA configuration
- learning rate
- physical batch size
- gradient accumulation
- maximum optimizer steps
- evaluation procedure
- experiment seeds

This allows observed differences to be interpreted primarily in relation to the
batch-order strategy.

## Dataset-Size Analysis

The Group 1 study includes controlled Dolly subsets of approximately:

- 1K examples
- 3K examples
- 5K examples

These settings allow the batching strategies to be examined across multiple
training-set sizes rather than relying on a single subset.

The same experimental interpretation should only be made after verifying that
the corresponding configurations follow the same intended protocol.

## Multi-Seed Design

The current experiment design uses three seeds:

- 13
- 21
- 42

A single run can be strongly affected by stochastic variation.

For that reason, conclusions should be based on behavior across seeds rather
than on the best individual run.

## Scope

The current research question is limited to the evaluated setup.

The study currently focuses on:

- instruction fine-tuning
- `google/flan-t5-small`
- LoRA-based parameter-efficient fine-tuning
- Dolly instruction data
- semantic grouping using `all-MiniLM-L6-v2`
- FAISS-based nearest-neighbor retrieval
- the four Group 1 batching strategies
- evaluation loss and ROUGE generation metrics

Results from this setup should not automatically be generalized to:

- substantially larger language models
- different fine-tuning algorithms
- different instruction datasets
- different embedding models
- different semantic grouping algorithms
- different optimization regimes

## Interpretation Principle

The objective of the research is not to prove that semantic batching is better.

The objective is to determine whether it provides a reliable measurable effect
under a controlled experimental setting.

A result showing no consistent improvement over random batching is therefore
also a valid research outcome.

## Versioning Note

This document describes the active thesis-v2 research question.

Earlier thesis experiments used materially different configurations and should
be treated as historical work rather than as direct evidence for the current
protocol unless explicitly identified and analyzed as such.
