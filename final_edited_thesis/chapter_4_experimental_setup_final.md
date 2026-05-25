Chapter 4
Experimental Setup

## 4.1 Experimental Pipeline

This chapter describes the implemented setup used to execute the experiments reported later in Chapter 5. The conceptual design has already been introduced in Chapter 3. The focus here is operational: dataset variants, preprocessing, model configuration, sampler implementation, experiment groups, evaluation procedures, and known setup caveats.

[Insert Figure 4.1 here: `../reports/thesis_figures/figure_4_1_experimental_pipeline.png`]

Figure 4.1. Implementation-specific experimental pipeline from Dolly subset preparation through tokenization, sampling, training, aggregation, and generation evaluation.

## 4.2 Dataset Variants and Preprocessing

The experiments use subsets of the Databricks Dolly instruction-following dataset. Three scales are used in the final study.

Table 4.1. Dataset scales used in the experiments.

| Dataset Variant | Train Size | Evaluation Size | Total Size | Main Use |
|---|---:|---:|---:|---|
| Dolly small 1k | 900 | 100 | 1,000 | Pilot and controlled 1k experiments |
| Dolly 3k | 2,700 | 300 | 3,000 | Intermediate-scale comparison |
| Dolly 5k | 4,500 | 500 | 5,000 | Main validation and curriculum experiments |

For the 3k and 5k variants, the raw Dolly source was deterministically shuffled with seed 42 and then split 90/10 into train and evaluation partitions. For the 1k runs, the executed split was also 900/100, but the raw-subset provenance is weaker than for the 3k and 5k variants and should be acknowledged as a reproducibility caveat.

Table 4.2. Preprocessing and tokenization settings.

| Setting | Dolly 1k | Dolly 3k / 5k |
|---|---|---|
| Input formatting | Instruction template with optional context block | Instruction only, or instruction plus context |
| Target text | Response | Response |
| Max input / label length | 256 | 512 |
| Truncation | True | True |
| Padding | False in the recovered 1k notebook path | `padding="max_length"` |
| Retained processed columns | Not fully persisted in a canonical config file | `input_ids`, `attention_mask`, `labels`, `raw_idx` |

## 4.3 Base Model and LoRA Configuration

All experiments use `google/flan-t5-small` as the base model. The final production scripts load the model through a sequence-to-sequence interface, while the local model configuration identifies the architecture as `T5ForConditionalGeneration`.

Table 4.3. Base model setup.

| Component | Value |
|---|---|
| Base model | `google/flan-t5-small` |
| Local model path | `models/flan-t5-small` |
| Architecture | `T5ForConditionalGeneration` |
| Model type | Encoder-decoder |
| Tokenizer family | T5 tokenizer |
| Maximum model length | 512 |

Fine-tuning is performed with LoRA in all production experiments.

Table 4.4. LoRA configuration.

| Hyperparameter | Value |
|---|---|
| Rank `r` | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | `q`, `v` |
| Task type | `SEQ_2_SEQ_LM` |
| Bias | `none` |

## 4.4 Training Configuration

The final scripted experiments use a shared Hugging Face training configuration across comparable blocks.

Table 4.5. Core training configuration.

| Setting | Value |
|---|---|
| Optimizer | AdamW Torch |
| Scheduler | Linear |
| Warmup steps | 0 |
| Per-device train batch size | 8 |
| Per-device eval batch size | 8 |
| Gradient accumulation steps | 4 |
| Effective train batch size | 32 |
| Weight decay | 0.01 |
| Logging steps | 10 |
| Evaluation strategy | Steps |
| Evaluation steps | 50 |
| Save strategy | Epoch |
| Save total limit | 2 |
| FP16 | False |
| BF16 | False |
| predict_with_generate during training | False |

Two-phase training is used throughout the controlled production comparisons. Phase 1 uses a learning rate of `7e-05`, and Phase 2 uses `5e-05`.

[Insert Figure 4.3 here: `../reports/thesis_figures/figure_4_3_two_phase_training_flow.png`]

Figure 4.3. Two-phase training flow showing phase-specific sampler modes, learning rates, adapter handoff, and the final checkpoint output.

Table 4.6. Training duration by experiment block.

| Experiment Block | Epochs per Phase |
|---|---:|
| `exp_012`-`exp_015` | 0.2 |
| `exp_018`-`exp_031` | 0.5 |
| `exp_032` | 1.0 |

## 4.5 Seed Handling and Reproducibility Caveat

The final experiment groups use three seeds: 13, 21, and 42. These values are passed to the custom samplers to control batch ordering. The persisted training arguments, however, also show `seed=42`, and no explicit global `set_seed`, `torch.manual_seed`, or equivalent call was recovered for the final production runs.

This matters for interpretation. The reported multi-seed variation clearly reflects differences in sampler ordering, but it may not reflect fully independent model-level random initialization in the strongest possible sense. The thesis therefore treats seed-based variability cautiously, especially where endpoint differences are very small.

## 4.6 Batch Scheduling Implementations

The final controlled comparison uses six production-implemented scheduling strategies:

- Random
- Grouped
- Grouped -> Random
- Random -> Grouped
- Easy -> Hard Length
- Hard -> Easy Length

Mixed batching was explored separately, but no final production mixed sampler or controlled mixed-batching configuration was found in the `configs`, `src`, or `scripts` paths used for the main experiment set.

[Insert Figure 4.4 here: `../reports/thesis_figures/figure_4_4_batch_scheduling_taxonomy.png`]

Figure 4.4. Taxonomy of the production-implemented batch scheduling strategies used in the final experiment matrix.

### 4.6.1 Random Batching

Random batching is implemented as the baseline sampler. It shuffles the processed training indices and emits contiguous batches of size 8.

### 4.6.2 Semantic Grouped Batching

Grouped batching uses an embedding-based nearest-neighbour pipeline. Each processed training row is aligned to a raw row, nearest neighbours are retrieved from the semantic index, and grouped mini-batches are assembled while tracking a global seen set to reduce repeated emission of examples.

[Insert Figure 4.2 here: `../reports/thesis_figures/figure_4_2_semantic_grouping_pipeline.png`]

Figure 4.2. Implementation-specific semantic grouping pipeline from raw examples through FAISS retrieval, alignment, sampler logic, and grouped mini-batch construction.

Table 4.7. Semantic grouping configuration.

| Dataset | Embedding Text | Embedding Model | FAISS Index | Similarity | Top-k | Batch Size |
|---|---|---|---|---|---:|---:|
| Dolly 1k | Instruction only | `all-MiniLM-L6-v2` | `IndexFlatIP` | Cosine via normalized inner product | 32 | 8 |
| Dolly 3k | Instruction + context | `all-MiniLM-L6-v2` | `IndexFlatIP` | Cosine via normalized inner product | 8 | 8 |
| Dolly 5k | Instruction + context | `all-MiniLM-L6-v2` | `IndexFlatIP` | Cosine via normalized inner product | 8 | 8 |

The 1k semantic index differs materially from the 3k and 5k indices. That inconsistency is a setup caveat and should be made explicit.

### 4.6.3 Two-Phase Semantic Curricula

The two-phase semantic curricula reuse the Phase 1 adapter in Phase 2 rather than restarting from the base model.

- Random -> Grouped:
  random batches in Phase 1, grouped batches in Phase 2.
- Grouped -> Random:
  grouped batches in Phase 1, random batches in Phase 2.

### 4.6.4 Length-Based Curricula

Length-based curricula are implemented through a curriculum sampler that orders samples by a length score. In the final implementation, the score is based on `len(input_ids)` plus the count of non-pad label tokens.

This setup has an important caveat. For the 3k and 5k processed datasets, the input side is padded to a fixed length of 512. That means the input-length contribution is effectively constant, so the ordering is driven mainly by the label side rather than by full prompt complexity. The length curriculum should therefore be interpreted as a label-length-oriented heuristic rather than as a direct measure of overall task difficulty.

## 4.7 Experiment Groups

Table 4.8. Main experiment groups used in the final thesis.

| Experiment IDs | Dataset Size | Methods | Training Duration |
|---|---|---|---|
| `exp_012`-`exp_015` | 1k | Random, Grouped, Grouped -> Random, Random -> Grouped | 0.2 epochs per phase |
| `exp_018`-`exp_021` | 1k | Random, Grouped, Grouped -> Random, Random -> Grouped | 0.5 epochs per phase |
| `exp_022`-`exp_025` | 3k | Random, Grouped, Grouped -> Random, Random -> Grouped | 0.5 epochs per phase |
| `exp_026`-`exp_029` | 5k | Random, Grouped, Grouped -> Random, Random -> Grouped | 0.5 epochs per phase |
| `exp_030`-`exp_031` | 5k | Easy -> Hard Length, Hard -> Easy Length | 0.5 epochs per phase |
| `exp_032` | 5k | Hard -> Easy Length | 1.0 epoch per phase |

## 4.8 Evaluation Setup

The evaluation procedure includes both loss-based and generation-based components.

Loss-based reporting uses the processed evaluation split associated with each dataset variant and aggregates:

- final train loss
- final eval loss
- generalization gap
- Phase 2 delta eval

Generation evaluation is available for the 5k batching and length-curriculum experiments. The evaluation script reconstructs the evaluation split using the raw data and a deterministic `train_test_split(test_size=0.1, seed=42)` procedure.

Table 4.9. Generation evaluation configuration.

| Setting | Value |
|---|---|
| Generation batch size | 8 |
| Max input length | 512 |
| Max new tokens | 128 |
| `do_sample` | False |
| ROUGE package | `evaluate.load("rouge")` |
| BERTScore package | `evaluate.load("bertscore")` |

## 4.9 Environment and Data-Reporting Caveats

Several caveats affect interpretation but do not invalidate the experiment set.

1. The 1k subset provenance is less fully documented than the 3k and 5k subsets.
2. The 1k semantic index differs from the 3k and 5k semantic indices in both text mode and effective neighbour width.
3. The seed protocol clearly changes sampler order, but may not fully reflect independent global random-state control.
4. The length curriculum is influenced strongly by label-length variation because the input side is padded to a fixed maximum length.
5. Aggregated generation summary CSVs conflict with some overlapping per-seed JSON files and should therefore be used consistently and cautiously.

## 4.10 Chapter Summary

This chapter has described the actual experimental configuration used in the thesis: the dataset variants, preprocessing choices, model and LoRA setup, training parameters, sampler implementations, experiment groups, and evaluation procedures. These implementation details provide the basis for the results reported in Chapter 5.
