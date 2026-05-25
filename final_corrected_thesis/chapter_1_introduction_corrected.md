Chapter 1
Introduction

## 1.1 Background and Motivation

Large language models (LLMs) have become central to current work in natural language processing because a single pre-trained model can be adapted to a wide range of downstream tasks. Pre-training provides broad linguistic and contextual knowledge, but it does not guarantee that a model will respond in a way that matches a user's instruction, output expectation, or task framing. Instruction fine-tuning addresses this gap by adapting a pre-trained model with instruction-response pairs rather than with a generic language-modelling objective alone.

This thesis examines one part of instruction fine-tuning that is usually treated as a routine implementation choice: the construction and ordering of mini-batches during training. In standard supervised fine-tuning pipelines, mini-batches are shuffled randomly. That choice is often reasonable, but it also leaves open a practical question. If the model is exposed to different local data structure during training, does the optimization trajectory change in a useful way?

The study focuses on two structured alternatives to fully random batching. The first is semantic grouping, where semantically related instructions are placed into the same mini-batch. The second is curriculum scheduling, where the training order is changed across phases or according to a length-based heuristic. The aim is not to introduce a new model architecture or a new fine-tuning method. The aim is to test whether a change in sample organization can alter instruction fine-tuning behaviour under controlled conditions.

[Insert Figure 1.1 here: `../reports/thesis_figures/figure_1_1_research_motivation_flow.png`]

Figure 1.1. Research motivation showing how this thesis narrows from instruction fine-tuning to the unexplored role of mini-batch scheduling.

## 1.2 Problem Statement

Instruction fine-tuning is commonly performed with randomly constructed mini-batches, yet the effect of batch composition and batch ordering on training dynamics is not well established. In particular, it is unclear whether semantically coherent batches or curriculum-style schedules can improve optimization behaviour, generalization, or generation quality relative to standard random batching.

This thesis addresses the following problem:

How does batch scheduling affect instruction fine-tuning of a large language model under a controlled LoRA-based training setup?

The problem is examined through direct comparison between random batching, semantic grouped batching, two-phase semantic curricula, and length-based curricula.

## 1.3 Research Objectives

The main objective of the thesis is to evaluate whether batch scheduling strategy affects instruction fine-tuning outcomes.

The specific objectives are:

1. To establish Random batching as a controlled baseline for instruction fine-tuning.
2. To implement semantic grouped batching using sentence embeddings and nearest-neighbour retrieval.
3. To compare two-phase semantic curricula in both directions: Grouped -> Random and Random -> Grouped.
4. To evaluate length-based curricula using Easy -> Hard and Hard -> Easy ordering.
5. To study these strategies across Dolly subsets of 1k, 3k, and 5k examples.
6. To compare methods using optimization metrics and generation metrics.
7. To assess whether observed differences are meaningful relative to seed variation.

## 1.4 Research Questions

The thesis is guided by the following questions:

1. Does semantic grouped batching improve instruction fine-tuning performance relative to Random batching?
2. Does the ordering of semantic structure across training phases affect learning behaviour?
3. Does Random remain a strong baseline when compared with more structured batching strategies?
4. Do length-based curricula provide a clearer signal than semantic grouping alone?
5. Are any observed gains large enough to be interpreted as practically meaningful under the tested setup?

## 1.5 Scope of the Study

The study is limited to supervised instruction fine-tuning of `google/flan-t5-small` with LoRA. The data source is the Databricks Dolly instruction-following dataset, evaluated through 1k, 3k, and 5k subsets. The final production strategies are:

- Random
- Grouped
- Grouped -> Random
- Random -> Grouped
- Easy -> Hard Length
- Hard -> Easy Length

The longer-training hard-to-easy run (`exp_032`) is analysed separately because it extends the training budget and is not a same-budget comparison with the standard 5k curriculum runs.

The thesis does not cover RLHF, inference-time retrieval augmentation, full-model pre-training, large-scale deployment, or human preference evaluation. It also does not treat mixed batching as part of the final controlled comparison; mixed batching appears only in exploratory material and is not part of the final production experiment matrix.

[Insert Figure 1.2 here: `../reports/thesis_figures/figure_1_2_research_scope_overview.png`]

Figure 1.2. Scope of the thesis, separating the investigated components from areas outside the study.

## 1.6 Significance of the Study

The study is relevant for three reasons. First, it examines a design choice that is present in every fine-tuning pipeline but is rarely isolated as an experimental variable. Second, it provides negative as well as positive evidence. If semantic grouping does not improve performance, that is still useful because it sets a realistic boundary on what batch-level semantic structure can achieve. Third, it tests whether curriculum-style ordering is more informative than static grouping, which has practical value for future work on efficient fine-tuning.

The contribution of the thesis is therefore not the proposal of a new training algorithm. It is a controlled empirical account of how batch scheduling behaves in instruction fine-tuning under modest resources, with explicit attention to weak effects, seed variation, and implementation caveats.

## 1.7 Thesis Structure

The remainder of the thesis is organised as follows.

- Chapter 2 reviews the literature on instruction fine-tuning, LoRA, mini-batch training, semantic similarity, curriculum learning, and evaluation metrics.
- Chapter 3 presents the conceptual methodology, including the strategy families, semantic grouping approach, curriculum designs, and evaluation framework.
- Chapter 4 describes the implementation-specific experimental setup, including datasets, preprocessing, model configuration, training settings, samplers, and experiment groups.
- Chapter 5 reports and analyses the results, with separate treatment of optimization outcomes, curriculum effects, generation metrics, and seed variance.
