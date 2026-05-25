Chapter 5
Results and Analysis

## 5.1 Overview

This chapter presents the results of the batch scheduling experiments described in Chapter 4. The analysis is organised around three questions. First, how do the compared strategies differ in endpoint optimization performance? Second, do semantic grouping or curriculum ordering provide any clear advantage over Random batching? Third, how large are the observed differences once seed variation and evaluation caveats are taken into account?

Unless stated otherwise, the canonical source for optimization results is `reports/master/master_summary_table.csv`. Per-seed stability comments are grounded in `reports/master/master_per_seed_results.csv`. Generation results are taken from the two aggregated generation summary CSVs for the 5k experiments, with the explicit caveat that those aggregates conflict with some overlapping per-seed JSON files.

## 5.2 Optimization Results by Experiment Block

### 5.2.1 Dolly 1k Short Training

Table 5.1. Dolly 1k short training comparison.

| Method | Experiment ID | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Generalization Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Random | `exp_012_random_only_multiseed` | 3 | 13,21,42 | 9.616399129231771 | 0.00012846433173675242 | 38.770100063747826 | -29.153700934516056 | -0.0013834635416666667 |
| Grouped | `exp_013_grouped_only_multiseed` | 3 | 13,21,42 | 9.61652692159017 | 0.00021044965469782815 | 38.722209506564674 | -29.105682584974502 | -0.0013230641682942708 |
| Grouped -> Random | `exp_014_grouped_to_random_multiseed` | 3 | 13,21,42 | 9.616591453552246 | 1.2615925364802315e-05 | 38.77133009168837 | -29.15473863813612 | -0.001311937967936198 |
| Random -> Grouped | `exp_015_random_to_grouped_multiseed` | 3 | 13,21,42 | 9.616386731465658 | 9.03947098320946e-05 | 38.722546895345054 | -29.106160163879395 | -0.0013634363810221355 |

Random -> Grouped gives the lowest mean eval loss in this block, while Grouped -> Random gives the highest. The absolute difference between best and worst is `0.000204722086588`. This is extremely small. The correct reading is that a possible ordering effect appears in the 1k short setting, but the margin is too narrow to justify a strong claim.

### 5.2.2 Dolly 1k Longer Training

Table 5.2. Dolly 1k longer training comparison.

| Method | Experiment ID | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Generalization Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Random | `exp_018_random_only_multiseed_long` | 3 | 13,21,42 | 9.611108779907227 | 0.00018671569539907427 | 38.760330539279515 | -29.149221759372285 | -0.004041353861490886 |
| Grouped | `exp_019_grouped_only_multiseed_long` | 3 | 13,21,42 | 9.611552556355795 | 0.00010812468756713155 | 38.722173394097226 | -29.11062083774143 | -0.0038159688313802085 |
| Grouped -> Random | `exp_020_grouped_to_random_multiseed_long` | 3 | 13,21,42 | 9.611759503682455 | 0.00021096115141597821 | 38.76272142198351 | -29.150961918301054 | -0.0037781397501627603 |
| Random -> Grouped | `exp_021_random_to_grouped_multiseed_long` | 3 | 13,21,42 | 9.611469268798828 | 0.0002447879672933447 | 38.722423977322045 | -29.11095470852322 | -0.003803253173828125 |

In the longer 1k setting, Random gives the lowest mean eval loss. Grouped -> Random again performs worst. Random -> Grouped remains better than Grouped -> Random, but it does not beat the Random baseline. This weakens any claim that the Random -> Grouped ordering is uniformly favourable.

### 5.2.3 Dolly 3k

Table 5.3. Dolly 3k comparison.

| Method | Experiment ID | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Generalization Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Random | `exp_022_random_only_multiseed_3k` | 3 | 13,21,42 | 11.423059145609537 | 0.0022963785111432176 | 45.43341644050539 | -34.010357294895854 | -0.03380934397379557 |
| Grouped | `exp_023_grouped_only_multiseed_3k` | 3 | 13,21,42 | 11.425052642822266 | 0.0006824924120630219 | 45.453193368837816 | -34.02814072601555 | -0.032903035481770836 |
| Grouped -> Random | `exp_024_grouped_to_random_multiseed_3k` | 3 | 13,21,42 | 11.424866358439127 | 0.001265824868793103 | 45.442404103833574 | -34.01753774539445 | -0.03327242533365885 |
| Random -> Grouped | `exp_025_random_to_grouped_multiseed_3k` | 3 | 13,21,42 | 11.425841649373373 | 0.000684287960140967 | 45.455506317375246 | -34.02966466800187 | -0.032347679138183594 |

At 3k, Random again gives the lowest mean eval loss, and Random -> Grouped becomes the weakest method in the block. The best-to-worst gap is `0.002782503763836`. This is still small in absolute terms, but it is large enough to show that the 1k short ordering pattern does not generalise cleanly.

### 5.2.4 Dolly 5k Main Comparison

Table 5.4. Dolly 5k main comparison.

| Method | Experiment ID | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Generalization Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Random | `exp_026_random_only_multiseed_5k` | 3 | 13,21,42 | 11.341803868611654 | 0.004030287838901957 | 45.12238576826355 | -33.7805818996519 | -0.07824929555257161 |
| Grouped | `exp_027_grouped_only_multiseed_5k` | 3 | 13,21,42 | 11.346779187520346 | 0.0025083908611327086 | 45.171566063249614 | -33.82478687572927 | -0.07535489400227864 |
| Grouped -> Random | `exp_028_grouped_to_random_multiseed_5k` | 3 | 13,21,42 | 11.344831148783365 | 0.0009995714022504269 | 45.13616887840307 | -33.7913377296197 | -0.07722759246826172 |
| Random -> Grouped | `exp_029_random_to_grouped_multiseed_5k` | 3 | 13,21,42 | 11.343379974365234 | 0.002895475059662966 | 45.15890305926543 | -33.81552308490019 | -0.07583808898925781 |

The 5k main block is the strongest core comparison because it uses the largest dataset scale among the directly comparable Random/Grouped/curriculum runs. Random again gives the lowest mean eval loss. Grouped performs worst, and the Grouped-versus-Random gap reaches `0.004975318908692`. Even here, however, the effect should be described as small rather than decisive.

[Insert Figure 5.1 here: `../reports/thesis_figures/figure_5_1_mean_eval_loss_by_block.png`]

Figure 5.1. Mean final evaluation loss across scheduling strategies and experiment blocks. Lower is better.

## 5.3 Random vs Semantic Grouping

The central hypothesis behind semantic grouping is that semantically coherent batches may produce more locally consistent gradient updates. The canonical results do not support a clear advantage for this idea.

Table 5.5. Grouped minus Random eval loss across directly comparable blocks.

| Block | Grouped Minus Random Eval Loss | Interpretation |
|---|---:|---|
| 1k short | 0.000127792358399 | Grouped is slightly worse |
| 1k long | 0.000443776448568 | Grouped is worse |
| 3k | 0.001993497212729 | Grouped is worse |
| 5k main | 0.004975318908692 | Grouped is worse |

Lower eval loss is better, so all four positive differences indicate that Grouped performs worse than Random in the directly comparable blocks. The pattern is consistent even though the effect size is modest.

This is an important negative result. It shows that increasing within-batch semantic coherence does not automatically improve instruction fine-tuning. A plausible explanation is that Random batching preserves useful task diversity inside each phase, whereas semantic grouping narrows the local gradient signal too strongly for this heterogeneous instruction dataset.

[Insert Figure 5.2 here: `../reports/thesis_figures/figure_5_2_difference_from_random_baseline.png`]

Figure 5.2. Difference in mean evaluation loss relative to the Random baseline. Negative values indicate improvement over Random.

The safest conclusion is therefore straightforward: Random batching remains a strong baseline, and semantic grouped batching does not consistently outperform it.

## 5.4 Curriculum Ordering and Phase-Wise Behaviour

### 5.4.1 Curriculum Ordering

The two-phase semantic curricula address a different question from pure grouping. Instead of asking only whether semantic structure helps, they ask whether the timing of semantic structure matters.

Table 5.6. Curriculum ordering comparison.

| Block | Random -> Grouped Mean Eval Loss | Grouped -> Random Mean Eval Loss | Better Ordering | Difference |
|---|---:|---:|---|---:|
| 1k short | 9.616386731465658 | 9.616591453552246 | Random -> Grouped | -0.000204722086588 |
| 1k long | 9.611469268798828 | 9.611759503682455 | Random -> Grouped | -0.000290234883627 |
| 3k | 11.425841649373373 | 11.424866358439127 | Grouped -> Random | 0.000975290934246 |
| 5k main | 11.343379974365234 | 11.344831148783365 | Random -> Grouped | -0.001451174418131 |

Random -> Grouped performs better in three of the four comparable blocks. The exception is the 3k block, where Grouped -> Random performs better. The margins are narrow throughout. This supports a cautious claim: there is a weak directional tendency in favour of Random -> Grouped, but not a uniform or decisive ordering advantage.

[Insert Figure 5.3 here: `../reports/thesis_figures/figure_5_3_curriculum_ordering_comparison.png`]

Figure 5.3. Comparison of two-phase semantic curriculum orderings across comparable experiment blocks.

### 5.4.2 Phase-2 Delta Eval

Phase-2 delta eval measures the change in evaluation loss during the second training phase. Negative values indicate that evaluation loss decreased during Phase 2. More negative values therefore indicate stronger Phase-2 improvement.

Table 5.7. Mean Phase-2 delta eval.

| Block | Method | Mean Phase2 Delta Eval |
|---|---|---:|
| 1k short | Random | -0.0013834635416666667 |
| 1k short | Grouped | -0.0013230641682942708 |
| 1k short | Grouped -> Random | -0.001311937967936198 |
| 1k short | Random -> Grouped | -0.0013634363810221355 |
| 1k long | Random | -0.004041353861490886 |
| 1k long | Grouped | -0.0038159688313802085 |
| 1k long | Grouped -> Random | -0.0037781397501627603 |
| 1k long | Random -> Grouped | -0.003803253173828125 |
| 3k | Random | -0.03380934397379557 |
| 3k | Grouped | -0.032903035481770836 |
| 3k | Grouped -> Random | -0.03327242533365885 |
| 3k | Random -> Grouped | -0.032347679138183594 |
| 5k main | Random | -0.07824929555257161 |
| 5k main | Grouped | -0.07535489400227864 |
| 5k main | Grouped -> Random | -0.07722759246826172 |
| 5k main | Random -> Grouped | -0.07583808898925781 |

All reported Phase-2 delta values in these blocks are negative. Phase 2 therefore improves evaluation loss in every case. Within each same-budget block, however, the differences between methods remain small. Random often has the strongest Phase-2 improvement among the Random/Grouped comparison blocks, which again reinforces the strength of the Random baseline.

[Insert Figure 5.4 here: `../reports/thesis_figures/figure_5_4_phase2_delta_eval_comparison.png`]

Figure 5.4. Mean evaluation-loss change during Phase 2. More negative values indicate larger Phase-2 improvement.

## 5.5 Length-Based Curriculum Results

### 5.5.1 Same-Budget Comparison

Table 5.8. Dolly 5k length-curriculum comparison.

| Method | Experiment ID | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Generalization Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Easy -> Hard Length | `exp_030_easy_to_hard_length_multiseed_5k` | 3 | 13,21,42 | 11.338643074035645 | 0.0012300206513966253 | 45.13238579118755 | -33.79374271715191 | -0.0795135498046875 |
| Hard -> Easy Length | `exp_031_hard_to_easy_length_multiseed_5k` | 3 | 13,21,42 | 11.337978998819986 | 0.0009772446107854716 | 45.12975311279297 | -33.791774113972984 | -0.08008988698323567 |

Hard -> Easy Length gives the lower mean eval loss by `0.000664075215659`. It also shows a slightly more negative Phase-2 delta than Easy -> Hard Length. The direction is therefore favourable to Hard -> Easy, but the same-budget margin remains very small.

### 5.5.2 Longer-Training Extension

Table 5.9. Extended hard-to-easy comparison.

| Method | Experiment ID | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Generalization Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Hard -> Easy Length | `exp_031_hard_to_easy_length_multiseed_5k` | 3 | 13,21,42 | 11.337978998819986 | 0.0009772446107854716 | 45.12975311279297 | -33.791774113972984 | -0.08008988698323567 |
| Hard -> Easy Length Longer Training | `exp_032_hard_to_easy_length_longer_training_5k` | 3 | 13,21,42 | 10.985996882120768 | 0.008129767881074827 | 44.02101001784593 | -33.03501313572516 | -0.2848030726114909 |

`exp_032` gives the lowest mean eval loss reported in the thesis: `10.985996882120768`. The absolute difference relative to the standard-budget hard-to-easy run is `0.351982116699218`. That result is clearly material, but it must not be treated as a same-budget comparison. The longer run changes both strategy exposure and training duration.

The length-based results also need an implementation caveat. Because the processed inputs are padded to a fixed maximum length, the input-side contribution to the curriculum score is effectively constant in the 3k and 5k processed datasets. The ordering is therefore driven mainly by label-side variation rather than by a full prompt-complexity notion of difficulty.

[Insert Figure 5.5 here: `../reports/thesis_figures/figure_5_5_length_curriculum_extended_training.png`]

Figure 5.5. Comparison of length-based curriculum strategies and the longer-training Hard -> Easy extension.

The most defensible reading is that length-based ordering appears more promising than semantic grouping, Hard -> Easy gives the clearest same-budget curriculum signal, and `exp_032` produces the strongest result overall while remaining a longer-training case.

## 5.6 Generation Quality

Generation quality is reported for the 5k batching and 5k length-curriculum experiments through the aggregated summary CSVs. These aggregates should be used consistently, but they must be read with the explicit caveat that some overlapping per-seed JSON files do not match them exactly.

Table 5.10. Generation metrics on 5k experiments.

| Method | Mean ROUGE-1 | Mean ROUGE-2 | Mean ROUGE-L | Mean ROUGE-Lsum | Mean BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| Random | 0.1332991221165225 | 0.05172597342614166 | 0.12528074807592213 | 0.1255176984312918 | 0.8358021107912063 |
| Grouped | 0.13515864447757928 | 0.05267632000153163 | 0.12741708266264154 | 0.12762222208271706 | 0.8365236309369406 |
| Grouped -> Random | 0.13573552053686477 | 0.05290762532282947 | 0.12741556756984396 | 0.1279101682552873 | 0.836249907930692 |
| Random -> Grouped | 0.13429778290121783 | 0.05139917328310489 | 0.12623164340206786 | 0.1263183377758188 | 0.8363362034161885 |
| Easy -> Hard Length | 0.1349535952911857 | 0.051761622149622204 | 0.12602883537238033 | 0.12640203541955244 | 0.8361860071023305 |
| Hard -> Easy Length | 0.13569431685742261 | 0.052930931439159314 | 0.12673431381120112 | 0.12715204647000258 | 0.8362131182750067 |

The metric leaders differ by metric:

- best ROUGE-1: Grouped -> Random (`0.13573552053686477`)
- best ROUGE-2: Hard -> Easy Length (`0.052930931439159314`)
- best ROUGE-L: Grouped (`0.12741708266264154`)
- best ROUGE-Lsum: Grouped -> Random (`0.1279101682552873`)
- best BERTScore F1: Grouped (`0.8365236309369406`)

The spreads remain narrow:

- ROUGE-1 spread: `0.00243639842034227`
- ROUGE-2 spread: `0.001531758156054424`
- ROUGE-L spread: `0.00213633458671941`
- ROUGE-Lsum spread: `0.0023924698239955`
- BERTScore F1 spread: `0.0007215201457343`

These values do not support a strong claim that batching strategy substantially changes output quality under the present automated evaluation setup.

[Insert Figure 5.6 here: `../reports/thesis_figures/figure_5_6_generation_metrics_comparison.png`]

Figure 5.6. Generation-quality metrics for 5k batching and length-curriculum experiments.

## 5.7 Seed Variance and Stability

The interpretation of the endpoint tables depends on the size of the seed variation. In several blocks, the standard deviation is of the same order as the method spread.

Table 5.11. Seed variance and stability summary.

| Block | Method | Std Eval Loss | Comment |
|---|---|---:|---|
| 1k short | Grouped | 0.00021044965469782815 | Full method spread is smaller than this standard deviation |
| 1k long | Random -> Grouped | 0.0002447879672933447 | Method differences remain very small in absolute terms |
| 3k | Random | 0.0022963785111432176 | Method spread is only modestly larger than the Random standard deviation |
| 5k main | Random | 0.004030287838901957 | Method spread is close to the largest standard deviation in the block |
| 5k length | Easy -> Hard Length | 0.0012300206513966253 | Same-budget method spread is smaller than both curriculum standard deviations |
| Extended hard-to-easy | Hard -> Easy Length Longer Training | 0.008129767881074827 | Larger standard deviation, but still far smaller than the mean gap relative to standard-budget hard-to-easy |

This limits the strength of the conclusions. The same-budget differences are often comparable to the observed seed variation. The strongest exception is `exp_032`, whose mean eval improvement over `exp_031` is much larger than its reported standard deviation, but that remains a longer-training rather than same-budget comparison.

[Insert Figure 5.7 here: `../reports/thesis_figures/figure_5_7_seed_variance_plot.png`]

Figure 5.7. Per-seed final evaluation loss showing seed-level variability across scheduling strategies.

## 5.8 Summary of Findings

Five findings can be stated safely.

1. Random batching remains a strong baseline.
   It gives the best mean eval loss in the 1k longer-training, 3k, and 5k main comparison blocks, and is effectively tied with the best result in the 1k short block.
2. Semantic grouped batching does not consistently outperform Random.
   In the directly comparable blocks, pure Grouped is worse than Random every time.
3. Curriculum ordering effects are present but small and mixed.
   Random -> Grouped performs better than Grouped -> Random in three of the four comparable blocks, but not in all of them, and not consistently against the Random baseline.
4. Hard -> Easy Length provides the clearest same-budget curriculum signal.
   It improves on Easy -> Hard Length at 5k, but only by a very small margin.
5. `exp_032` gives the best reported result overall, but it uses longer training and should not be presented as a same-budget comparison.

Taken together, the results suggest that batch scheduling can influence training behaviour, but its effect on final performance is limited under the tested conditions. Random batching remains difficult to beat, semantic grouping alone is not sufficient, and length-based ordering appears more promising than semantic grouping while still requiring conservative interpretation.
