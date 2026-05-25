# 1. Files Inspected

| Path | What It Contains | Status |
|---|---|---|
| `D:\Working\llm-batching-research\reports\master\master_summary_table.csv` | Aggregated optimization results for `exp_012` to `exp_032`, including mean/std eval loss, train loss, generalization gap, and mean phase-2 delta eval. | Canonical for optimization results |
| `D:\Working\llm-batching-research\reports\master\master_per_seed_results.csv` | Per-seed optimization results for each experiment, with phase-1 and phase-2 losses and final endpoint metrics. | Canonical for per-seed optimization results |
| `D:\Working\llm-batching-research\reports\master\master_summary.json` | JSON mirror of the master optimization summary table. | Secondary mirror |
| `D:\Working\llm-batching-research\reports\generation_eval_summaries\generation_metrics_summary_table.csv` | Aggregated generation metrics for `exp_026` to `exp_029`. | Canonical for generation metrics for the 5k batching-method block |
| `D:\Working\llm-batching-research\reports\generation_eval_5k\generation_metrics_summary_table.csv` | Aggregated generation metrics for `exp_030` and `exp_031`. | Canonical for generation metrics for the 5k length-curriculum block |
| `D:\Working\llm-batching-research\reports\generation_eval_summaries\exp_026_random_only_multiseed_5k\seed_013\generation_metrics.json` and matching `seed_021` / `seed_042` JSONs | Per-seed generation metrics used to check whether the aggregated CSV values match the underlying JSONs for `exp_026` to `exp_029`. | Conflicting check source |
| `D:\Working\llm-batching-research\reports\generation_eval_5k\exp_030_easy_to_hard_length_multiseed_5k\seed_013\generation_metrics.json` and matching `seed_021` / `seed_042` JSONs | Per-seed generation metrics used to check whether the aggregated CSV values match the underlying JSONs for `exp_030` and `exp_031`. | Conflicting check source |
| `D:\Working\llm-batching-research\reports\archive\exp_018_to_exp_021_summary_table.csv` | Older archived 1k-long optimization summary block. | Archived and conflicting with current master summary for `exp_018` |
| `D:\Working\llm-batching-research\reports\plots\combined_plot_data.csv` | Plot-data snapshot used by older plotting scripts. Contains only `16` rows and only the blocks `1k_short`, `1k_long`, `3k_long`, and `5k_long`. | Stale / incomplete |

# 2. Canonical Result Sources

1. Optimization results:
   Use `D:\Working\llm-batching-research\reports\master\master_summary_table.csv`.

2. Per-seed optimization results:
   Use `D:\Working\llm-batching-research\reports\master\master_per_seed_results.csv`.

3. Generation metrics:
   Use both:
   `D:\Working\llm-batching-research\reports\generation_eval_summaries\generation_metrics_summary_table.csv`
   and
   `D:\Working\llm-batching-research\reports\generation_eval_5k\generation_metrics_summary_table.csv`.
   Caveat: these aggregated CSVs do not fully match all overlapping per-seed `generation_metrics.json` files. The mismatch is material for `exp_026`, `exp_030`, and `exp_031`, and negligible for `exp_027` to `exp_029`.

4. Archived results:
   Treat `D:\Working\llm-batching-research\reports\archive\*.csv` and `*.json` as provenance only, not as chapter-table sources. In particular, `reports\archive\exp_018_to_exp_021_summary_table.csv` conflicts with the current master summary for `exp_018_random_only_multiseed_long`.

5. Plots:
   Regenerate all Chapter 5 figures from the canonical CSVs above.

Files that should not be used as primary Chapter 5 sources:

- `D:\Working\llm-batching-research\reports\plots\combined_plot_data.csv`
  Reason: incomplete; it omits the length-curriculum rows and `exp_032`.
- `D:\Working\llm-batching-research\reports\plots\plot_cross_block_eval_trends.png`
  Reason: stale and incomplete.
- `D:\Working\llm-batching-research\reports\plots\plot_5k_full_curriculum_comparison_eval_loss.png`
  Reason: misleading; it does not faithfully include the full 5k curriculum set.
- Archived summary CSVs
  Reason: they are not the current canonical aggregates.

# 3. Optimization Results Tables

## 3.1 1k Short Training Comparison: `exp_012` to `exp_015`

| Method | Experiment ID | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Generalization Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Random | `exp_012_random_only_multiseed` | 3 | 13,21,42 | 9.616399129231771 | 0.00012846433173675242 | 38.770100063747826 | -29.153700934516056 | -0.0013834635416666667 |
| Grouped | `exp_013_grouped_only_multiseed` | 3 | 13,21,42 | 9.61652692159017 | 0.00021044965469782815 | 38.722209506564674 | -29.105682584974502 | -0.0013230641682942708 |
| Grouped->Random | `exp_014_grouped_to_random_multiseed` | 3 | 13,21,42 | 9.616591453552246 | 1.2615925364802315e-05 | 38.77133009168837 | -29.15473863813612 | -0.001311937967936198 |
| Random->Grouped | `exp_015_random_to_grouped_multiseed` | 3 | 13,21,42 | 9.616386731465658 | 9.03947098320946e-05 | 38.722546895345054 | -29.106160163879395 | -0.0013634363810221355 |

## 3.2 1k Long Training Comparison: `exp_018` to `exp_021`

| Method | Experiment ID | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Generalization Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Random | `exp_018_random_only_multiseed_long` | 3 | 13,21,42 | 9.611108779907227 | 0.00018671569539907427 | 38.760330539279515 | -29.149221759372285 | -0.004041353861490886 |
| Grouped | `exp_019_grouped_only_multiseed_long` | 3 | 13,21,42 | 9.611552556355795 | 0.00010812468756713155 | 38.722173394097226 | -29.11062083774143 | -0.0038159688313802085 |
| Grouped->Random | `exp_020_grouped_to_random_multiseed_long` | 3 | 13,21,42 | 9.611759503682455 | 0.00021096115141597821 | 38.76272142198351 | -29.150961918301054 | -0.0037781397501627603 |
| Random->Grouped | `exp_021_random_to_grouped_multiseed_long` | 3 | 13,21,42 | 9.611469268798828 | 0.0002447879672933447 | 38.722423977322045 | -29.11095470852322 | -0.003803253173828125 |

## 3.3 3k Comparison: `exp_022` to `exp_025`

| Method | Experiment ID | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Generalization Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Random | `exp_022_random_only_multiseed_3k` | 3 | 13,21,42 | 11.423059145609537 | 0.0022963785111432176 | 45.43341644050539 | -34.010357294895854 | -0.03380934397379557 |
| Grouped | `exp_023_grouped_only_multiseed_3k` | 3 | 13,21,42 | 11.425052642822266 | 0.0006824924120630219 | 45.453193368837816 | -34.02814072601555 | -0.032903035481770836 |
| Grouped->Random | `exp_024_grouped_to_random_multiseed_3k` | 3 | 13,21,42 | 11.424866358439127 | 0.001265824868793103 | 45.442404103833574 | -34.01753774539445 | -0.03327242533365885 |
| Random->Grouped | `exp_025_random_to_grouped_multiseed_3k` | 3 | 13,21,42 | 11.425841649373373 | 0.000684287960140967 | 45.455506317375246 | -34.02966466800187 | -0.032347679138183594 |

## 3.4 5k Main Comparison: `exp_026` to `exp_029`

| Method | Experiment ID | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Generalization Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Random | `exp_026_random_only_multiseed_5k` | 3 | 13,21,42 | 11.341803868611654 | 0.004030287838901957 | 45.12238576826355 | -33.7805818996519 | -0.07824929555257161 |
| Grouped | `exp_027_grouped_only_multiseed_5k` | 3 | 13,21,42 | 11.346779187520346 | 0.0025083908611327086 | 45.171566063249614 | -33.82478687572927 | -0.07535489400227864 |
| Grouped->Random | `exp_028_grouped_to_random_multiseed_5k` | 3 | 13,21,42 | 11.344831148783365 | 0.0009995714022504269 | 45.13616887840307 | -33.7913377296197 | -0.07722759246826172 |
| Random->Grouped | `exp_029_random_to_grouped_multiseed_5k` | 3 | 13,21,42 | 11.343379974365234 | 0.002895475059662966 | 45.15890305926543 | -33.81552308490019 | -0.07583808898925781 |

## 3.5 5k Length Curriculum Comparison: `exp_030` to `exp_031`

| Method | Experiment ID | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Generalization Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Easy->Hard Length | `exp_030_easy_to_hard_length_multiseed_5k` | 3 | 13,21,42 | 11.338643074035645 | 0.0012300206513966253 | 45.13238579118755 | -33.79374271715191 | -0.0795135498046875 |
| Hard->Easy Length | `exp_031_hard_to_easy_length_multiseed_5k` | 3 | 13,21,42 | 11.337978998819986 | 0.0009772446107854716 | 45.12975311279297 | -33.791774113972984 | -0.08008988698323567 |

## 3.6 Extended Hard-to-Easy Comparison: `exp_031` vs `exp_032`

| Method | Experiment ID | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Generalization Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Hard->Easy Length | `exp_031_hard_to_easy_length_multiseed_5k` | 3 | 13,21,42 | 11.337978998819986 | 0.0009772446107854716 | 45.12975311279297 | -33.791774113972984 | -0.08008988698323567 |
| Hard->Easy Length Longer Training | `exp_032_hard_to_easy_length_longer_training_5k` | 3 | 13,21,42 | 10.985996882120768 | 0.008129767881074827 | 44.02101001784593 | -33.03501313572516 | -0.2848030726114909 |

# 4. Best/Worst Method Per Block

| Block | Best Method by Mean Eval Loss | Best Mean Eval Loss | Worst Method by Mean Eval Loss | Worst Mean Eval Loss | Absolute Difference | Percentage Difference | Practical Reading |
|---|---|---:|---|---:|---:|---:|---|
| 1k short | Random->Grouped | 9.616386731465658 | Grouped->Random | 9.616591453552246 | 0.000204722086588 | 0.002129% vs best | Extremely small |
| 1k long | Random | 9.611108779907227 | Grouped->Random | 9.611759503682455 | 0.000650723775228 | 0.006771% vs best | Extremely small |
| 3k | Random | 11.423059145609537 | Random->Grouped | 11.425841649373373 | 0.002782503763836 | 0.024359% vs best | Small |
| 5k main | Random | 11.341803868611654 | Grouped | 11.346779187520346 | 0.004975318908692 | 0.043867% vs best | Small, still narrow |
| 5k length curriculum | Hard->Easy Length | 11.337978998819986 | Easy->Hard Length | 11.338643074035645 | 0.000664075215659 | 0.005857% vs best | Extremely small |
| Extended hard-to-easy | Hard->Easy Length Longer Training | 10.985996882120768 | Hard->Easy Length | 11.337978998819986 | 0.351982116699218 | 3.104452% vs standard hard->easy | Material, but not same-budget comparable |

Reading:

- For the same-budget comparisons, the optimization differences are narrow in every block.
- The only clearly material gap is `exp_032` versus `exp_031`, but that is a longer-training comparison rather than a clean same-budget method comparison.

# 5. Random Baseline Comparison

Lower eval loss is better. Negative difference versus Random means improvement over Random. Positive difference means worse performance than Random.

| Block | Method | Eval Loss Minus Random | Improves or Worsens vs Random | Interpretation |
|---|---|---:|---|---|
| 1k short | Grouped | 0.000127792358399 | Worsens | Slightly worse than Random |
| 1k short | Grouped->Random | 0.000192324320475 | Worsens | Slightly worse than Random |
| 1k short | Random->Grouped | -0.000012397766113 | Improves | Only a marginal improvement over Random |
| 1k long | Grouped | 0.000443776448568 | Worsens | Worse than Random |
| 1k long | Grouped->Random | 0.000650723775228 | Worsens | Worst of the 1k-long block relative to Random |
| 1k long | Random->Grouped | 0.000360488891601 | Worsens | Worse than Random |
| 3k | Grouped | 0.001993497212729 | Worsens | Worse than Random |
| 3k | Grouped->Random | 0.001807212829590 | Worsens | Worse than Random, but closer than pure Grouped |
| 3k | Random->Grouped | 0.002782503763836 | Worsens | Worst against Random in the 3k block |
| 5k main | Grouped | 0.004975318908692 | Worsens | Clearly worse than Random within this block |
| 5k main | Grouped->Random | 0.003027280171711 | Worsens | Worse than Random |
| 5k main | Random->Grouped | 0.001576105753580 | Worsens | Closest alternative to Random in the 5k main block, but still worse |

Block-level reading:

- 1k short: Random is effectively tied with Random->Grouped; the apparent improvement is only `-0.000012397766113`.
- 1k long: Random is best.
- 3k: Random is best.
- 5k main: Random is best.

# 6. Semantic Grouping Finding

- Semantic grouped batching does not consistently outperform Random. In these canonical optimization results, pure Grouped is worse than Random in all four directly comparable blocks:
  - 1k short: `+0.000127792358399`
  - 1k long: `+0.000443776448568`
  - 3k: `+0.001993497212729`
  - 5k main: `+0.004975318908692`
- Random remains a strong baseline. It is the best method by mean eval loss in `1k_long`, `3k_long`, and `5k_long`, and is effectively tied with the best method in `1k_short`.
- The absolute Grouped-versus-Random gap grows with dataset scale, but the effect size is still small in absolute terms.
- The differences are often comparable to seed variation:
  - 1k short: Grouped-minus-Random (`0.000127792358399`) is smaller than the Grouped std (`0.00021044965469782815`).
  - 3k: Grouped-minus-Random (`0.001993497212729`) is smaller than the Random std (`0.0022963785111432176`).
  - 5k main: Grouped-minus-Random (`0.004975318908692`) is only modestly larger than the Random std (`0.004030287838901957`).
- Safest thesis claim:
  Pure semantic grouped batching did not improve final optimization performance over the Random baseline in the canonical experiments. Random remained a strong and often best-performing baseline, and any semantic-grouping effect should be described as weak and inconsistent rather than as a robust gain.

# 7. Curriculum Ordering Finding

| Block | Random->Grouped Mean Eval Loss | Grouped->Random Mean Eval Loss | Better Ordering | Difference (`Random->Grouped` minus `Grouped->Random`) |
|---|---:|---:|---|---:|
| 1k short | 9.616386731465658 | 9.616591453552246 | Random->Grouped | -0.000204722086588 |
| 1k long | 9.611469268798828 | 9.611759503682455 | Random->Grouped | -0.000290234883627 |
| 3k | 11.425841649373373 | 11.424866358439127 | Grouped->Random | 0.000975290934246 |
| 5k main | 11.343379974365234 | 11.344831148783365 | Random->Grouped | -0.001451174418131 |

Reading:

- `Random->Grouped` is better in three of the four blocks: `1k_short`, `1k_long`, and `5k_long`.
- `Grouped->Random` is better in `3k_long`.
- This is not a consistent result pattern.
- The margins remain small in every block.
- `Random->Grouped` also does not consistently beat the Random baseline:
  - it slightly improves on Random in `1k_short`
  - it is worse than Random in `1k_long`, `3k_long`, and `5k_long`

Thesis-safe conclusion:

- There is a weak directional tendency in favor of `Random->Grouped` over `Grouped->Random`, which is compatible with an exploration-then-refinement interpretation.
- The evidence is not strong enough for a broad claim that exploration->refinement is consistently superior.
- The safest framing is that ordering effects exist, but they are small and not uniformly beneficial relative to the Random baseline.

# 8. Phase-2 Delta Eval Analysis

Negative phase-2 delta eval means evaluation loss decreased during phase 2. More negative values indicate larger phase-2 improvement.

| Block | Method | Mean Phase2 Delta Eval | Interpretation |
|---|---|---:|---|
| 1k short | Random | -0.0013834635416666667 | Strongest phase-2 improvement in the 1k-short block |
| 1k short | Grouped | -0.0013230641682942708 | Negative improvement, but slightly weaker than Random |
| 1k short | Grouped->Random | -0.001311937967936198 | Negative improvement, weakest in the block |
| 1k short | Random->Grouped | -0.0013634363810221355 | Negative improvement, close to Random |
| 1k long | Random | -0.004041353861490886 | Strongest phase-2 improvement in the 1k-long block |
| 1k long | Grouped | -0.0038159688313802085 | Negative improvement, weaker than Random |
| 1k long | Grouped->Random | -0.0037781397501627603 | Negative improvement, weakest in the block |
| 1k long | Random->Grouped | -0.003803253173828125 | Negative improvement, weaker than Random |
| 3k | Random | -0.03380934397379557 | Strongest phase-2 improvement in the 3k block |
| 3k | Grouped | -0.032903035481770836 | Negative improvement, weaker than Random |
| 3k | Grouped->Random | -0.03327242533365885 | Negative improvement, second strongest in the block |
| 3k | Random->Grouped | -0.032347679138183594 | Negative improvement, weakest in the block |
| 5k main | Random | -0.07824929555257161 | Strongest phase-2 improvement in the 5k main block |
| 5k main | Grouped | -0.07535489400227864 | Negative improvement, weaker than Random |
| 5k main | Grouped->Random | -0.07722759246826172 | Negative improvement, close to Random |
| 5k main | Random->Grouped | -0.07583808898925781 | Negative improvement, weaker than Random |
| 5k length curriculum | Easy->Hard Length | -0.0795135498046875 | Strong phase-2 improvement |
| 5k length curriculum | Hard->Easy Length | -0.08008988698323567 | Slightly stronger than Easy->Hard |
| Extended hard-to-easy | Hard->Easy Length Longer Training | -0.2848030726114909 | Much larger phase-2 improvement, but confounded by longer training budget |

Reading:

- Phase 2 helps in every block and for every method because all mean phase-2 delta values are negative.
- The magnitude of phase-2 improvement grows sharply with the larger-scale and longer-budget runs.
- Within a given same-budget block, the between-method phase-2 differences are still small.

# 9. Length Curriculum Results

## Exact comparisons

| Comparison | Exact Mean Eval Losses | Exact Difference | Reading |
|---|---|---:|---|
| Easy->Hard vs Hard->Easy | Easy->Hard: `11.338643074035645`; Hard->Easy: `11.337978998819986` | `-0.000664075215659` for Hard->Easy minus Easy->Hard | Hard->Easy is better, but by a very small same-budget margin |
| Hard->Easy standard vs Hard->Easy longer training | Standard: `11.337978998819986`; Longer: `10.985996882120768` | `-0.351982116699218` for longer minus standard | Longer training is much better, but not same-budget comparable |
| Easy->Hard vs Hard->Easy phase-2 delta | Easy->Hard: `-0.0795135498046875`; Hard->Easy: `-0.08008988698323567` | `-0.00057633717854817` for Hard->Easy minus Easy->Hard | Hard->Easy also shows slightly stronger phase-2 improvement |
| Hard->Easy standard vs longer phase-2 delta | Standard: `-0.08008988698323567`; Longer: `-0.2848030726114909` | `-0.20471318562825523` for longer minus standard | Longer training drives a much larger phase-2 gain |

## Implementation caveat

- The length-curriculum sampler computes a curriculum score as `len(input_ids) + nonpad_label_count`.
- The processed datasets are padded with `padding="max_length"` at `512`.
- As a result, the input-length component is effectively constant across samples, so the ordering is driven mainly by non-padded label length rather than by full prompt-plus-target length.
- This weakens any strong interpretation that the sampler is measuring overall example difficulty.

## Thesis-safe interpretation

- Within the same 5k training budget, `Hard->Easy Length` is better than `Easy->Hard Length`, but the difference is very small: `0.000664075215659` in mean eval loss.
- The stronger signal is the longer-training extension: `exp_032` reaches `10.985996882120768`, which is substantially lower than `11.337978998819986` for standard-budget `Hard->Easy Length`.
- The longer-training result should not be presented as a clean same-budget strategy win; it is a budget-extension result.
- The cautious thesis claim is that length-based ordering appears more promising than semantic grouping, `Hard->Easy` shows the clearest optimization signal within that family, but the standard-budget difference remains small and the sampler implementation caveat limits strong claims about difficulty ordering.

# 10. Generation Metrics Results

Source caveat:

- The combined generation table below uses the two aggregated summary CSVs exactly as requested.
- These aggregated CSVs conflict with overlapping per-seed `generation_metrics.json` files.
- The mismatch is effectively rounding-level for `exp_027` to `exp_029`, but is material enough to mention for `exp_026`, `exp_030`, and `exp_031`.

| exp_id | Strategy / Method | Mean ROUGE-1 | Std ROUGE-1 | Mean ROUGE-2 | Std ROUGE-2 | Mean ROUGE-L | Std ROUGE-L | Mean ROUGE-Lsum | Std ROUGE-Lsum | Mean BERTScore F1 | Std BERTScore F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `exp_026_random_only_multiseed_5k` | Random | 0.1332991221165225 | 0.0020164915902323 | 0.05172597342614166 | 0.001973474178185322 | 0.12528074807592213 | 0.001879402357167189 | 0.1255176984312918 | 0.001713859683307618 | 0.8358021107912063 | 0.0004622980477068232 |
| `exp_027_grouped_only_multiseed_5k` | Grouped | 0.13515864447757928 | 0.000821310043745659 | 0.05267632000153163 | 0.0008599748859510768 | 0.12741708266264154 | 0.0008997847037711997 | 0.12762222208271706 | 0.0009568022221241892 | 0.8365236309369406 | 0.00029145341957466175 |
| `exp_028_grouped_to_random_multiseed_5k` | Grouped->Random | 0.13573552053686477 | 0.0006310815717993919 | 0.05290762532282947 | 0.0007638326220628706 | 0.12741556756984396 | 0.00040348389836700736 | 0.1279101682552873 | 0.00047559550741240033 | 0.836249907930692 | 0.00025796425857546053 |
| `exp_029_random_to_grouped_multiseed_5k` | Random->Grouped | 0.13429778290121783 | 0.0009638989478806322 | 0.05139917328310489 | 0.0006283945874624924 | 0.12623164340206786 | 0.0008984621429260791 | 0.1263183377758188 | 0.0008879813126434798 | 0.8363362034161885 | 0.0004125665273872965 |
| `exp_030_easy_to_hard_length_multiseed_5k` | Easy->Hard Length | 0.1349535952911857 | 0.0022750204251505116 | 0.051761622149622204 | 0.0019208392324216834 | 0.12602883537238033 | 0.0020777819486155744 | 0.12640203541955244 | 0.002185766668899262 | 0.8361860071023305 | 0.00028109593218276314 |
| `exp_031_hard_to_easy_length_multiseed_5k` | Hard->Easy Length | 0.13569431685742261 | 0.001371932873660898 | 0.052930931439159314 | 0.0014885726613267686 | 0.12673431381120112 | 0.0014466042675307454 | 0.12715204647000258 | 0.0011848330035887271 | 0.8362131182750067 | 0.00021941865358977844 |

Metric leaders from the aggregated CSVs:

- Best ROUGE-1:
  `exp_028_grouped_to_random_multiseed_5k` with `0.13573552053686477`
- Best ROUGE-2:
  `exp_031_hard_to_easy_length_multiseed_5k` with `0.052930931439159314`
- Best ROUGE-L:
  `exp_027_grouped_only_multiseed_5k` with `0.12741708266264154`
- Best ROUGE-Lsum:
  `exp_028_grouped_to_random_multiseed_5k` with `0.1279101682552873`
- Best BERTScore F1:
  `exp_027_grouped_only_multiseed_5k` with `0.8365236309369406`

Spread across the six methods:

- ROUGE-1 spread: `0.00243639842034227`
- ROUGE-2 spread: `0.001531758156054424`
- ROUGE-L spread: `0.00213633458671941`
- ROUGE-Lsum spread: `0.0023924698239955`
- BERTScore F1 spread: `0.0007215201457343`

Thesis-safe reading:

- The generation metrics vary only slightly across methods.
- No method dominates all generation metrics.
- The aggregated generation evaluation does not support strong claims of substantial output-quality improvement from any single batching strategy.

# 11. Seed Variance and Stability

Per-seed optimization results in `master_per_seed_results.csv` confirm that each reported experiment aggregate is based on exactly three seeds: `13`, `21`, and `42`.

| Block | Method | Std Eval Loss | Comment |
|---|---|---:|---|
| 1k short | Grouped | 0.00021044965469782815 | The full method spread in the block (`0.000204722086588`) is smaller than this std. Seed variation is comparable to or larger than method variation. |
| 1k long | Random->Grouped | 0.0002447879672933447 | The block spread (`0.000650723775228`) is larger than any single std, but still very small in absolute terms. |
| 3k | Random | 0.0022963785111432176 | The block spread (`0.002782503763836`) is only modestly larger than the Random std. Method differences are close to seed-scale variation. |
| 5k main | Random | 0.004030287838901957 | The block spread (`0.004975318908692`) is again close to the largest std. Rankings should be interpreted cautiously. |
| 5k length curriculum | Easy->Hard Length | 0.0012300206513966253 | The same-budget method spread (`0.000664075215659`) is smaller than both curriculum std values. |
| Extended hard-to-easy | Hard->Easy Length Longer Training | 0.008129767881074827 | Larger seed spread than the standard hard-to-easy run, but still far smaller than the `0.351982116699218` mean-eval improvement over standard-budget hard-to-easy. |

Safest interpretation:

- Seed variation is often comparable to, and sometimes larger than, the observed same-budget method differences.
- That is especially true for `1k_short` and the `5k` length-curriculum comparison.
- The stronger `exp_032` result stands out beyond its own seed std, but it remains a longer-training comparison rather than a clean strategy-only effect.

# 12. Figures Recommended for Chapter 5

Do not use the old plot PNGs directly. Regenerate all Chapter 5 figures from the canonical CSVs.

| Figure Title | Source File | X-axis | Y-axis | Grouping | Plot Type | Zoom Y-axis? | Caption | Caveat |
|---|---|---|---|---|---|---|---|---|
| Mean Eval Loss by Method Across Blocks | `reports\master\master_summary_table.csv` | Method | Mean Eval Loss | Block | Faceted dot plot or faceted bar plot | Yes, per panel | Mean final evaluation loss for each scheduling strategy within each experiment block. | Small differences can look exaggerated; annotate exact values. |
| Difference from Random Baseline | `reports\master\master_summary_table.csv` | Block | Eval Loss Minus Random | Method excluding Random | Diverging dot plot or diverging bar plot | Yes, centered around 0 | Difference in mean eval loss relative to the Random baseline within each block. Negative values indicate improvement over Random. | Most deltas are small; the zero line must be visually prominent. |
| Curriculum Ordering Comparison | `reports\master\master_summary_table.csv` | Block | Mean Eval Loss | `Grouped->Random` vs `Random->Grouped` | Dumbbell plot | Yes | Direct comparison of the two semantic curriculum orderings across the four comparable blocks. | The ordering effect is not consistent across all blocks. |
| Phase-2 Delta Eval Comparison | `reports\master\master_summary_table.csv` | Method | Mean Phase2 Delta Eval | Block | Faceted dot plot | Yes, per panel | Mean evaluation-loss change during phase 2 for each strategy and block. More negative values indicate larger phase-2 gains. | Cross-block magnitudes differ sharply; faceting is safer than one global axis. |
| Length Curriculum and Extended Training Comparison | `reports\master\master_summary_table.csv` | Strategy / Experiment | Mean Eval Loss | Standard-budget vs longer-training hard-to-easy | Paired dot plot or compact bar plot | Yes | Comparison of easy-to-hard and hard-to-easy length curricula, with the longer-training hard-to-easy extension shown separately. | The longer-training result is not a same-budget comparison. |
| Generation Metrics Comparison | `reports\generation_eval_summaries\generation_metrics_summary_table.csv` and `reports\generation_eval_5k\generation_metrics_summary_table.csv` | Strategy / Experiment | Metric Score | Metric | Faceted dot plot | Yes, per metric | Aggregated ROUGE and BERTScore results for the 5k batching and length-curriculum experiments. | Aggregated CSVs conflict with some per-seed JSONs. |
| Seed Variance Plot | `reports\master\master_per_seed_results.csv` and `reports\master\master_summary_table.csv` | Method | Final Eval Loss | Seed and block | Strip plot with mean marker and std overlay | Yes, per panel | Per-seed final evaluation loss with block-wise means and variability overlays. | Only three seeds per experiment; do not overstate stability claims. |

# 13. Thesis-Safe Claims

## Supported Claims

- Random is a strong optimization baseline and is the best method by mean eval loss in `1k_long`, `3k_long`, and `5k_long`.
- Pure semantic grouped batching does not beat Random in any of the four directly comparable optimization blocks.
- All methods show negative mean phase-2 delta eval, so phase 2 consistently reduces eval loss further.
- Within the standard 5k length-curriculum comparison, Hard->Easy Length has lower mean eval loss than Easy->Hard Length.
- The longer-training Hard->Easy Length run (`exp_032`) achieves substantially lower mean eval loss than the standard-budget Hard->Easy Length run (`exp_031`).

## Tentative Claims

- `Random->Grouped` appears somewhat more favorable than `Grouped->Random` across most blocks, but the evidence is mixed and the margins are small.
- Length-based ordering appears more promising than semantic grouping in the 5k experiments, especially for Hard->Easy Length, but the same-budget effect is small.
- Generation metrics vary slightly by strategy, but the differences are narrow and do not point to a single dominant method.

## Claims to Avoid

- Do not claim that semantic grouping consistently improves optimization performance.
- Do not claim that any same-budget method difference is statistically significant; no statistical test was inspected.
- Do not claim substantial generation-quality improvement from batching strategy alone.
- Do not present `exp_032` as a clean strategy win over `exp_031`; it is a longer-training result.
- Do not describe the length curriculum as a validated difficulty curriculum without mentioning that the current implementation is driven mainly by label length under fixed-length input padding.

# 14. Final Chapter 5 Narrative Plan

## 5.1 Overview

State that Chapter 5 reports optimization and generation outcomes for the final production strategies across the 1k, 3k, and 5k experiment blocks. Clarify that optimization results come from the canonical master summary CSV and generation results come from the two aggregated generation summary CSVs, with caveats noted where needed.

## 5.2 Optimization Results

Present the core endpoint tables for `1k_short`, `1k_long`, `3k_long`, `5k_long`, and the 5k length-curriculum block. Emphasize exact mean eval loss rankings while noting that most same-budget differences are narrow.

## 5.3 Random vs Semantic Grouping

Use direct deltas from the Random baseline to show that pure Grouped does not outperform Random in the canonical runs. Frame Random as a robust baseline and avoid claiming strong semantic-grouping gains.

## 5.4 Curriculum Ordering

Compare `Grouped->Random` and `Random->Grouped` across the four comparable blocks. Note the weak tendency toward `Random->Grouped` while stressing that the effect is inconsistent and small.

## 5.5 Length-Based Curriculum

Compare Easy->Hard Length and Hard->Easy Length at the same 5k budget, then discuss the longer-training Hard->Easy extension separately. Include the implementation caveat that the current sampler is driven mainly by label length because of fixed-length padded inputs.

## 5.6 Generation Quality

Present the combined generation-metric table for `exp_026` to `exp_031` and identify metric leaders cautiously. State that differences are small and that the aggregated generation CSVs conflict with some per-seed JSONs.

## 5.7 Stability and Seed Variance

Use the per-seed results and std columns to show that seed variation is often comparable to the observed same-budget method spread. This section should temper over-interpretation of very small ranking differences.

## 5.8 Summary of Findings

Close the chapter with the safest claims: Random is a strong baseline, semantic grouping does not show robust gains, curriculum ordering effects are weak, Hard->Easy Length is the most promising same-budget curriculum signal, and the strongest improvement appears in the longer-training hard-to-easy extension rather than in a clean same-budget strategy shift.
