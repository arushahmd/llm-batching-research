# Experiment Protocol

This document records the canonical human-readable protocol for the active
thesis-v2 Group 1 experiments.

The machine-readable source for the current canonical 1K experiment is:

```text
configs/group1/dolly_1k.yaml
```

The purpose of this document is to make the experimental design easy to audit
without requiring readers to reconstruct the protocol from notebooks or source
code.

## Protocol Status

The Dolly 1K configuration is currently the canonical Group 1 template.

The 3K and 5K notebooks are preserved in the repository, but their final YAML
configurations should be created only after they are reviewed against the same
intended protocol.

Accordingly, this document treats the 1K configuration as the authoritative
machine-readable reference for thesis-v2 at this stage.

## Experimental Objective

The experiment compares four physical mini-batch ordering strategies while
keeping the remaining training setup controlled.

The strategies are:

1. `random`
2. `grouped`
3. `grouped_to_random`
4. `random_to_grouped`

Each strategy is evaluated with three experiment seeds:

```text
13
21
42
```

This produces:

```text
4 strategies × 3 seeds = 12 runs
```

for each dataset-size setting when the same protocol is applied.

## Base Model

```text
google/flan-t5-small
```

The model is loaded fresh for every individual strategy/seed run.

Parameter-efficient fine-tuning is performed using LoRA.

## Dataset

Dataset:

```text
databricks/databricks-dolly-15k
```

Source split:

```text
train
```

Current canonical subset size:

```text
1,000 examples
```

The dataset is shuffled before subset selection using:

```text
shuffle_seed = 42
```

The first 1,000 examples after this deterministic shuffle are selected.

## Train / Evaluation Split

The selected subset is divided using:

```text
eval_ratio = 0.10
split_seed = 42
```

For the 1K setting this yields approximately:

```text
900 training examples
100 evaluation examples
```

The same split is reused across batching strategies and experiment seeds.

## Dolly Input Formatting

Each Dolly example is converted into an instruction-tuning input and target.

When context is present:

```text
Instruction: <instruction>
Context: <context>
```

When context is absent:

```text
Instruction: <instruction>
```

The target is the Dolly response field.

## Tokenization

Input maximum length:

```text
256 tokens
```

Target maximum length:

```text
128 tokens
```

Both input and target sequences use:

```text
truncation = True
padding = "max_length"
```

### Reproducibility Note on Target Padding

The completed current Group 1 notebooks keep padded target token IDs directly
in the label sequence.

They do not replace target padding positions with `-100`.

The active thesis-v2 implementation intentionally preserves this behavior so
that the curated code reflects the protocol that was actually executed.

Changing target-padding masking would alter the training objective and should
therefore be treated as a new protocol requiring an explicit rerun.

## LoRA Configuration

The current Group 1 LoRA configuration is:

```text
r = 8
alpha = 16
dropout = 0.05
target_modules = ["q", "v"]
task_type = SEQ_2_SEQ_LM
```

A fresh base model is loaded and LoRA adapters are attached for every
strategy/seed run.

## Training Configuration

Physical training batch size:

```text
3
```

Gradient accumulation steps:

```text
2
```

Effective examples contributing to one optimizer update:

```text
3 × 2 = 6
```

Evaluation batch size:

```text
2
```

Maximum optimizer steps:

```text
300
```

Learning rate:

```text
0.003
```

Logging interval:

```text
50 steps
```

Mixed precision:

```text
fp16 = False
```

The current notebook protocol does not perform periodic evaluation or checkpoint
saving during these Group 1 runs.

Training arguments use:

```text
eval_strategy = "no"
save_strategy = "no"
report_to = "none"
predict_with_generate = True
```

Evaluation is performed manually after training.

## Physical Batch Count

The batching strategy constructs the complete physical-batch order required for
the run before training begins.

The number of physical batches is:

```text
max_steps × gradient_accumulation_steps
```

For the current protocol:

```text
300 × 2 = 600 physical batches
```

With a physical batch size of 3, the fixed-order sampler contains:

```text
600 × 3 = 1,800 example positions
```

Training examples may therefore appear multiple times across a run.

## Semantic Embedding Configuration

Embedding model:

```text
all-MiniLM-L6-v2
```

Current embedding mode:

```text
embed_instruction_only = False
```

This means semantic text is constructed from:

```text
instruction + context
```

when context exists, otherwise from the instruction alone.

Embeddings are converted to `float32` and L2-normalized.

## FAISS Similarity Search

The semantic index uses:

```text
faiss.IndexFlatIP
```

Because embeddings are L2-normalized, inner-product search corresponds to
cosine-similarity ranking.

Semantic neighbor pool:

```text
top_k = 8
```

## Random Batching Strategy

For every physical mini-batch:

1. a seeded NumPy random generator is used;
2. `batch_size` training indices are sampled;
3. sampling is without replacement inside that physical batch.

Examples may repeat across different physical batches.

The complete order is deterministic for a fixed experiment seed.

## Grouped Batching Strategy

Grouped batching constructs each physical batch around an anchor example.

The procedure is:

1. create a pool containing all training indices;
2. shuffle the anchor pool using the experiment seed;
3. select the next anchor;
4. query the FAISS index with the anchor embedding;
5. request `top_k + 1` candidates;
6. remove the anchor itself;
7. retain the first `top_k` semantic neighbors;
8. randomly sample up to `batch_size - 1` neighbors;
9. combine the anchor and selected neighbors;
10. use a random training example as fallback only if the batch is still
    incomplete.

When all anchors have been consumed, the anchor pool is reshuffled and reused.

## Curriculum Strategies

The physical-batch sequence is divided into two phases using:

```text
half = n_batches // 2
```

### Grouped → Random

The first phase uses grouped batching.

The second phase uses random batching with:

```text
second_phase_seed = seed + 1000
```

### Random → Grouped

The first phase uses random batching.

The second phase uses grouped batching with:

```text
second_phase_seed = seed + 1000
```

If the number of physical batches is odd, the second phase receives the
remaining batch.

For the current 600-batch protocol the split is:

```text
300 physical batches
300 physical batches
```

## Fixed-Order Training

The generated sequence of dataset indices is passed to a custom
`FixedOrderSampler`.

`OrderedTrainer`, derived from Hugging Face `Seq2SeqTrainer`, overrides the
training DataLoader so that examples are consumed in exactly the predefined
order.

The DataLoader uses:

```text
batch_size = per_device_train_batch_size
sampler = FixedOrderSampler(...)
collate_fn = DataCollatorForSeq2Seq(...)
drop_last = True
```

This fixed-order mechanism is essential because batching strategy is the
experimental variable.

## Experiment Seed Handling

Before loading the fresh model for an individual run, the current protocol sets:

```text
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
```

Additional deterministic CUDA or cuDNN settings are not part of the currently
executed Group 1 protocol and are therefore not silently introduced into the
curated implementation.

## Evaluation Loss

After training completes:

```text
trainer.evaluate()
```

is used to obtain the final evaluation loss on the held-out evaluation split.

No intermediate evaluation schedule is required by the current Group 1
protocol.

## Generation Evaluation

After training, the model generates outputs for the evaluation examples.

Generation batch size:

```text
64
```

Maximum newly generated tokens:

```text
128
```

which corresponds to the configured maximum target length.

Predictions are decoded with special tokens removed.

References are the original Dolly `target_text` values from the held-out
evaluation set.

## Generation Metrics

The verified Group 1 generation metrics are:

```text
ROUGE-1 F1
ROUGE-2 F1
ROUGE-L F1
```

ROUGE scoring uses stemming.

The per-example F1 scores are averaged across the evaluation split.

### BERTScore

BERTScore is not calculated by the current verified Group 1 implementation.

Historical notebook comments or imports referring to BERTScore should not be
interpreted as evidence that BERTScore was part of the executed thesis-v2
metric pipeline.

## Result Record

Each individual run produces a lightweight result record containing:

```text
strategy
seed
eval_loss
rouge1
rouge2
rougeL
```

The current runner writes raw rerun outputs under the ignored local
`outputs/` directory.

Only reviewed and validated lightweight results should eventually be promoted
to the version-controlled `results/` directory.

## Dataset-Size Extension

The repository also contains Group 1 notebooks for Dolly 3K and 5K subsets.

Before creating their canonical YAML files or publishing aggregate comparisons,
their settings should be checked against this protocol.

Only verified differences such as dataset subset size should be carried into
those configurations unless a deliberate protocol change is documented.

## Legacy Protocol Separation

Earlier thesis experiments used materially different settings, including
different LoRA parameters, batch sizes, learning rates, training duration, and
semantic-neighbor settings.

Those experiments belong to the preserved legacy research version.

They should not be merged into thesis-v2 result tables or conclusions as though
they were generated under the current protocol.

## Change-Control Principle

The active repository follows this rule:

```text
completed protocol → preserve for reproducibility
methodology improvement → document as a new protocol and rerun
```

Code cleanup should not silently change the scientific meaning of an already
completed experiment.
