# Research Question

## Primary Question

Under a matched instruction fine-tuning budget, how do **mini-batch
composition** and **training-example order** affect optimization and generation
quality?

## Group 1

Does semantic mini-batch construction, or transitioning between semantic and
random batching, produce a reliable advantage over random batching?

Strategies: `random`, `grouped`, `grouped_to_random`, `random_to_grouped`.

## Group 2

Does fully ordering training examples by input length change fine-tuning
outcomes, and does the curriculum direction matter?

Directions: `easy_to_hard`, `hard_to_easy`.

Difficulty is the full tokenized formatted instruction+context length before
training-time truncation.

## Outcomes

- evaluation loss
- ROUGE-1
- ROUGE-2
- ROUGE-L
- consistency across seeds

BERTScore is not part of the verified pipeline.

## Interpretation

The study does not assume structured batching or curriculum learning is better.
Null, mixed, scale-dependent, and metric-dependent results are valid outcomes.
Findings apply to the tested model, data, scales, budget, and evaluation setup.
