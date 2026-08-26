# Methodology

## Data

Dolly is deterministically shuffled, subset to 1K/3K/5K, formatted as
instruction plus optional context, and split 90/10.

## Fixed-Order Training

Both experiment groups construct the complete dataset-index sequence before
training. A custom sampler and `OrderedTrainer` consume examples in exactly that
order.

## Group 1 — Semantic Composition

Instruction+context embeddings are produced with `all-MiniLM-L6-v2`, normalized,
and indexed with FAISS. The study compares random batching, semantic grouping,
and two semantic/random phase orders.

## Group 2 — Length Ordering

Difficulty is the full tokenizer-output length of instruction+context before
training truncation. Examples are sorted ascending for Easy → Hard and
descending for Hard → Easy. The order cycles when necessary.

## Training

Every method/seed run starts from a fresh FLAN-T5-small model with the same
LoRA and optimization setup.

## Evaluation

Final evaluation loss and ROUGE-1/2/L are reported per seed and as multi-seed
mean/sample-standard-deviation summaries.

## Artifact Curation

Executed notebooks remain research records. Stable behavior is represented in
source modules/configs, while reviewed metrics and figures live under
`results/`. Public Group 2 notebooks have outputs stripped to avoid exposing a
stale post-run output in the raw 1K notebook.
