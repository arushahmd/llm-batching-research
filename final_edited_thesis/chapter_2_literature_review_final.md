Chapter 2
Literature Review

## 2.1 Introduction

This chapter reviews the literature most directly related to the thesis: instruction fine-tuning, parameter-efficient adaptation, mini-batch training, semantic similarity, curriculum learning, and automatic evaluation for generated text. The purpose is not to restate the full history of these topics. It is to identify the specific research gap addressed here: the role of mini-batch construction and scheduling during instruction fine-tuning.

Instruction fine-tuning has been studied extensively at the levels of model design, data curation, alignment, and scale. By contrast, mini-batch scheduling is often treated as a background implementation detail. That asymmetry matters. If batching strategy changes the local gradients seen by the model, then it may influence convergence, generalization, or generation quality without changing the dataset or the architecture.

## 2.2 Large Language Models and Instruction Fine-Tuning

Modern LLMs are built largely on Transformer-based architectures, which replaced recurrence with self-attention and made large-scale sequence modelling practical. The T5 family extended this line of work by casting many tasks into a unified text-to-text framework, an especially natural fit for instruction-response learning. Flan-T5 further demonstrated that instruction tuning across diverse tasks can improve generalisation and instruction-following behaviour.

This line of work provides the immediate foundation for the present study. The thesis uses `google/flan-t5-small` precisely because it is already aligned with the instruction-tuning setting, computationally manageable, and suitable for controlled comparisons. The question is not whether instruction fine-tuning works in general. That is well established. The question is whether training outcomes change when the organization of mini-batches changes.

## 2.3 Parameter-Efficient Fine-Tuning

Repeated full-parameter fine-tuning is expensive, particularly when a study compares multiple strategies, dataset scales, and seeds. Parameter-efficient fine-tuning addresses that problem by adapting only a small subset of parameters or by introducing lightweight trainable modules while keeping most of the pre-trained model fixed.

LoRA is one of the most widely used methods in this category. By introducing trainable low-rank matrices into selected layers, it reduces memory and training cost while retaining a strong basis for controlled comparison. In the present thesis, LoRA is not itself the object of investigation. It is the practical mechanism that makes repeated multi-seed comparison feasible.

## 2.4 Mini-Batch Training as an Experimental Variable

Mini-batch training is standard practice in neural network optimisation. Instead of computing a gradient over the full dataset, training proceeds with smaller subsets of examples, which provide stochastic gradient estimates and make training computationally feasible.

In most supervised fine-tuning pipelines, mini-batches are formed randomly. Random batching avoids strong ordering bias, mixes heterogeneous samples, and is often a sensible default. In instruction-tuning datasets, that diversity may be particularly useful because batches can contain different task types, response styles, and contextual structures.

What is less established is whether mini-batch construction should remain entirely random once semantic structure is available. A semantically coherent batch may offer a more locally consistent learning signal. At the same time, that coherence reduces within-batch diversity. The literature gives reasons to consider both possibilities, but it does not provide a clear answer for instruction fine-tuning.

## 2.5 Semantic Similarity and Retrieval-Based Grouping

Semantic grouping depends on two technical ingredients: sentence-level embeddings and efficient similarity search. Sentence-BERT made semantically meaningful sentence embeddings practical for retrieval and clustering tasks, while FAISS provided a scalable way to perform nearest-neighbour search over dense vector collections.

Together, these methods make semantic mini-batch construction straightforward in engineering terms. Instruction texts can be embedded, indexed, and grouped by nearest-neighbour structure. What remains uncertain is whether that grouping improves fine-tuning. The existence of a semantic retrieval mechanism does not by itself justify a claim about optimization benefit. That is an empirical question.

## 2.6 Curriculum Learning and Training Order

Curriculum learning introduced the broader idea that the order of training examples can influence learning. In the classical formulation, models are exposed to easier examples first and harder examples later. Subsequent work has shown that ordering effects can be task-dependent, and in some settings reverse curricula can also be useful.

This literature is relevant because it shifts the focus from static composition to temporal structure. In the present study, that idea is adapted to mini-batch scheduling in two ways. First, semantic structure can be introduced earlier or later through two-phase schedules. Second, example order can be approximated through a length-based heuristic. These designs make it possible to test not only whether structure matters, but when structure might matter.

## 2.7 Evaluation of Instruction-Tuned Models

Evaluation in instruction fine-tuning usually combines optimization-based and generation-based criteria. Training loss and evaluation loss provide direct information about fit to the observed data, while the generalization gap offers a simple view of how training and evaluation behaviour diverge. In a two-phase setup, the change in evaluation loss during the second phase is also informative because it captures the contribution of the phase transition itself.

Automatic generation metrics add another perspective. ROUGE captures lexical overlap, while BERTScore uses contextual embeddings to compare generated and reference text more semantically. Neither metric is a complete substitute for human judgment, but together they provide a practical basis for consistent comparison when manual evaluation is outside scope.

## 2.8 Position of the Present Study

The present thesis sits at the intersection of several established areas: instruction fine-tuning, parameter-efficient fine-tuning, semantic retrieval, curriculum learning, and automatic evaluation. Its contribution is narrower than any one of those fields. It does not propose a new Transformer architecture, a new PEFT method, or a new evaluation metric. Instead, it isolates batch scheduling as the central research variable.

[Insert Figure 2.1 here: `../reports/thesis_figures/figure_2_1_position_in_existing_work.png`]

Figure 2.1. Positioning of the present study relative to existing work in instruction fine-tuning, curriculum learning, PEFT, and semantic retrieval.

## 2.9 Research Gap

The literature supports the plausibility of structured batching, but it does not establish that such structure should help instruction fine-tuning in practice. Existing work has concentrated more on model scale, task coverage, alignment, and data selection than on the composition of individual mini-batches or the timing of structure across training phases.

That gap can be stated precisely: instruction fine-tuning lacks careful empirical comparison of mini-batch construction and scheduling strategies under a controlled, multi-seed setup. This thesis addresses that gap by comparing Random batching with semantic grouping, two-phase semantic curricula, and length-based curricula across multiple dataset scales.

## 2.10 Chapter Summary

The literature shows why this problem is worth testing. Instruction fine-tuning is well established, LoRA makes repeated experiments feasible, semantic embeddings and FAISS make semantic grouping technically practical, and curriculum learning provides a framework for thinking about training order. What remains unresolved is whether these ingredients improve instruction fine-tuning once they are translated into mini-batch scheduling strategies.

The next chapter sets out the conceptual methodology used to answer that question.
