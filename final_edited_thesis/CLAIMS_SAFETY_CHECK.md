# Claims Safety Check

## Supported Claims

- Random batching remains a strong baseline in the final experiment set.
- Pure semantic grouped batching does not outperform Random in the directly comparable optimization blocks.
- Curriculum ordering effects are small and mixed rather than uniformly favourable.
- Hard -> Easy Length provides the clearest same-budget curriculum signal in the 5k length-curriculum comparison.
- `exp_032` produces the best reported optimization result overall, but it does so under a longer training budget.
- Generation metric differences across the 5k methods are small.
- Seed variation is often comparable to same-budget method differences.
- Phase 2 reduces evaluation loss in every reported experiment block.

## Tentative Claims

- Random -> Grouped shows a weak directional tendency over Grouped -> Random in the two-phase semantic curriculum comparison.
- Length-based ordering appears more promising than semantic grouping in this study.
- Hard -> Easy ordering may be a more useful curriculum direction than Easy -> Hard under the tested setup.

These claims were kept explicitly cautious in the edited chapters and were not framed as definitive or general laws.

## Claims Removed or Softened

- Strong language suggesting that semantic grouping improves instruction fine-tuning was removed.
- Any wording that implied statistical significance was removed.
- Any wording that treated mixed batching as part of the final production comparison was removed.
- Any wording that treated `exp_032` as a same-budget result was removed.
- Strong claims that the length curriculum reflects true task difficulty were softened because the implemented score is influenced heavily by fixed-length padded inputs and label-side variation.
- Broad claims about generation-quality gains were softened because metric spreads are narrow and source caveats remain.

## Places Still Needing Supervisor Confirmation

- Whether the final thesis should keep chapter-local tables in full or move some dense result tables to an appendix during final typesetting.
- Whether the bibliography should be chapter-local or consolidated at thesis level in the final submission format.
- Whether the institution's preferred template expects figure placeholders to remain in draft form or to be replaced by inserted figures before supervisor review.
- Whether the seed-handling caveat in Chapter 4 should be stated exactly as written or shortened for the final submission version.
- Whether the length-curriculum implementation caveat should remain in the main text or be shortened and moved to a limitations subsection.
