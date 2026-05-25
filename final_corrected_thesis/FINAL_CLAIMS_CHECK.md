## Final Claims Check

### Required confirmations

- No semantic batching improvement claim remains.
  Confirmed. The corrected chapters retain the position that Random remains a strong baseline and that semantic grouped batching does not consistently outperform Random.

- `exp_032` is clearly treated as longer-training.
  Confirmed. Chapter 5 states that `exp_032` gives the best reported result overall but must not be read as a same-budget comparison. Figure 5.5 now separates it visually as a longer-training extension.

- Mixed batching is not treated as a production strategy.
  Confirmed. Chapter 4 states that mixed batching was exploratory only and that no final production mixed sampler or controlled mixed-batching configuration was part of the main experiment set.

- Generation metric differences are described as small.
  Confirmed. Chapter 5 keeps the interpretation that ROUGE and BERTScore spreads are narrow and do not support a strong claim of substantial quality improvement.

- Seed variation caveat is preserved.
  Confirmed. Chapters 4 and 5 both retain the caveat that only three seeds were used per experiment and that same-budget differences are often comparable to observed seed variation.

### Supported claims still present

- Random batching remains a strong baseline.
- Semantic grouped batching does not consistently outperform Random.
- Curriculum ordering effects are small and mixed.
- Hard -> Easy Length gives the clearest same-budget curriculum signal.
- `exp_032` gives the strongest reported result overall, but under a longer training budget.
- Generation metric differences are small under the automated evaluation setup used here.

### Claims removed or softened

- Any reading that would imply a clear optimization advantage for semantic grouping over Random remains excluded.
- Any wording that could treat `exp_032` as directly comparable to the standard-budget 5k runs has been kept out.
- Figure-related wording that could overstate generation-metric differences or seed-stability evidence has been kept cautious.

### Places that may still need supervisor confirmation

- Whether Chapter 2 should keep its own reference list or fold into a single thesis-wide bibliography in the final Word/PDF version.
- Whether the department template expects figure caveats in captions, notes below captions, or surrounding body text rather than inline Markdown prose.
