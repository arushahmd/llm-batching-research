Chapter 3
Methodology

## 3.1 Research Design

The study is a controlled comparative experiment on instruction fine-tuning. The independent variable is the batch scheduling strategy. The base model family, fine-tuning approach, dataset source, and evaluation protocol are held constant within each comparable experiment block. This design makes it possible to attribute observed differences, however small, to changes in batch construction or ordering rather than to changes in architecture or data source.

The methodological emphasis is empirical rather than algorithmic. The thesis does not propose a new model, a new embedding method, or a new optimizer. It asks whether the organisation of training samples changes the behaviour of an otherwise fixed instruction fine-tuning pipeline.

[Insert Figure 3.1 here: `../reports/thesis_figures/figure_3_1_methodology_workflow.png`]

Figure 3.1. Overall methodology workflow for evaluating batch scheduling strategies in instruction fine-tuning.

## 3.2 Strategy Families

The final production comparison includes three strategy families.

1. Static strategies:
   Random and Grouped.
2. Two-phase semantic curricula:
   Grouped -> Random and Random -> Grouped.
3. Length-based curricula:
   Easy -> Hard Length and Hard -> Easy Length.

Mixed batching appears in exploratory notebook material, but it is not part of the final controlled production comparison and is not treated as a main thesis result.

[Insert Figure 3.2 here: `../reports/thesis_figures/figure_3_2_batch_scheduling_strategies.png`]

Figure 3.2. Production strategy families evaluated in the thesis: static batching, two-phase curricula, and length-based curricula.

## 3.3 Semantic Mini-Batch Construction

Semantic grouping is based on a simple idea: instructions that are close in embedding space may provide a more locally coherent training signal when placed into the same mini-batch. To test that idea, the methodology treats semantic grouping as a reproducible pipeline rather than as an informal heuristic.

The conceptual steps are:

1. Convert instruction text into dense sentence embeddings.
2. Normalize the embeddings for similarity search.
3. Build a nearest-neighbour structure over the embedding set.
4. Select anchor examples and retrieve nearby neighbours.
5. Form grouped mini-batches from the anchor-neighbour sets.

This design increases within-batch semantic coherence, but it also reduces within-batch diversity. The thesis therefore treats semantic grouping as a hypothesis to be tested rather than as an assumed improvement over Random batching.

[Insert Figure 3.3 here: `../reports/thesis_figures/figure_3_3_semantic_minibatch_construction.png`]

Figure 3.3. Schematic of semantic mini-batch construction using sentence embeddings, FAISS nearest-neighbour retrieval, and anchor-based grouping.

## 3.4 Curriculum Scheduling Designs

The study distinguishes between composition and ordering. Static grouping changes which examples appear together. Curriculum scheduling changes when a particular type of batch structure is presented during training.

Two semantic curriculum directions are evaluated:

- Random -> Grouped:
  the model first sees diverse batches, then more semantically focused batches.
- Grouped -> Random:
  the model first sees semantically structured batches, then later returns to random mixing.

Two length-based curricula are also evaluated:

- Easy -> Hard Length:
  shorter examples first, longer examples later.
- Hard -> Easy Length:
  longer examples first, shorter examples later.

These curricula are designed to test whether training order influences the optimization path, even when the model and data source remain fixed.

[Insert Figure 3.4 here: `../reports/thesis_figures/figure_3_4_curriculum_scheduling_designs.png`]

Figure 3.4. Curriculum scheduling designs evaluated in the study, spanning semantic ordering and length-based ordering.

## 3.5 Experimental Staging

The methodological workflow progresses in stages rather than in a single monolithic comparison.

1. Early controlled 1k experiments establish that the scheduling implementations behave as intended.
2. Longer 1k runs test whether short-run observations persist under a larger training budget.
3. The 3k and 5k blocks test whether any pattern survives scaling.
4. The final 5k length-curriculum runs examine whether a difficulty-style ordering signal is more promising than semantic grouping alone.

This staged design matters because a method that appears promising in a short pilot may weaken or reverse when the data scale or training duration changes.

## 3.6 Multi-Seed Protocol

Key experiments are repeated with seeds 13, 21, and 42. The purpose of the multi-seed design is not to claim statistical significance from a small sample, but to reduce the risk of treating a single training run as representative. Means and standard deviations are therefore reported alongside endpoint metrics.

The study treats seed variation as part of the substantive interpretation. Where method differences are of similar scale to seed variation, the results are read cautiously.

## 3.7 Evaluation Framework

The evaluation framework combines optimization-based and generation-based criteria.

Optimization metrics:

- final train loss
- final eval loss
- generalization gap
- phase-wise delta eval during Phase 2

Generation metrics:

- ROUGE-1
- ROUGE-2
- ROUGE-L
- ROUGE-Lsum
- BERTScore F1

This combination is necessary because endpoint loss alone does not fully describe instruction-following behaviour, while automatic generation metrics alone do not explain the training dynamics that produced a result.

## 3.8 Validity and Reliability Considerations

Several choices are intended to improve reliability.

- Random batching is treated as a serious baseline rather than as a weak default.
- Comparable blocks keep the model family, fine-tuning method, and reporting pipeline fixed.
- Multi-seed aggregation reduces the weight of any single run.
- Multiple dataset scales reduce the chance that a conclusion depends entirely on a pilot setting.
- Both optimization and generation metrics are reported, with caveats where the generation aggregates conflict with some per-seed JSON files.

The methodology therefore supports modest claims well. It is not designed to justify sweeping claims from very small numerical gaps.

## 3.9 Chapter Summary

This chapter has defined the conceptual methodology of the study: a controlled comparison of batch scheduling strategies for instruction fine-tuning, centred on semantic grouping, curriculum ordering, and length-based scheduling. The next chapter moves from this conceptual design to the implementation-specific setup used in the actual experiment runs.
