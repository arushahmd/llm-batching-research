# Format and Style Report

## Reference Format Followed

The reference thesis PDF was used as a formatting model only, not as a content source. The edited chapters follow the same broad submission pattern requested for the final draft:

- chapter title on a separate line in the form `Chapter N`
- chapter name on the next line
- numbered subsections within each chapter
- chapter-based figure numbering
- chapter-based table numbering
- figure captions placed immediately below figure placeholders
- table captions presented consistently above the corresponding table

Because the reference PDF text extraction was limited, the strongest formatting guidance came from the visible chapter structure in the reference file together with the formatting rules you specified in the task prompt.

## Chapter Structure Used

- `chapter_1_introduction_final.md`
  Focused on background, problem statement, objectives, questions, scope, significance, and thesis structure.
- `chapter_2_literature_review_final.md`
  Focused on instruction fine-tuning, LoRA, mini-batch training, semantic grouping, curriculum learning, evaluation, and the research gap.
- `chapter_3_methodology_final.md`
  Kept conceptual and research-design oriented.
- `chapter_4_experimental_setup_final.md`
  Kept implementation-specific, covering datasets, preprocessing, model setup, training setup, samplers, evaluation configuration, and setup caveats.
- `chapter_5_results_analysis_final.md`
  Kept results-driven and cautious, with all numerical values preserved from the canonical result sources.

## Figure and Table Placement Summary

Figures were placed according to `reports/thesis_figures/FIGURE_INDEX.md` and inserted as explicit placeholders with captions:

- Chapter 1: Figures 1.1 and 1.2
- Chapter 2: Figure 2.1
- Chapter 3: Figures 3.1 to 3.4
- Chapter 4: Figures 4.1 to 4.4
- Chapter 5: Figures 5.1 to 5.7

Tables were added where they support exposition directly:

- Chapter 4 contains setup/configuration tables only.
- Chapter 5 contains result and interpretation tables only.

## Writing and Editing Changes Applied

- Repetition between Chapters 1 to 4 was reduced.
- Chapter 3 was rewritten to remain conceptual rather than implementation-heavy.
- Chapter 4 was rewritten to hold the implementation-specific material.
- Chapter 5 was tied directly to the canonical result summaries and rewritten in a more cautious analytical style.
- Unsupported or overstated claims from the draft language were removed or softened.
- Mixed batching was excluded from the final production comparison narrative.

## Remaining Formatting Issues

- The output is in Markdown, so final page-level formatting still needs to be applied in the submission format used for the full thesis document.
- Citation formatting and final bibliography integration still need supervisor or author-side typesetting work if the thesis is being assembled in Word or LaTeX.
- Figure placeholders currently reference the generated thesis figure assets; final thesis assembly may require replacing placeholders with the actual inserted figures.
- Table layout may need manual width adjustment in the final submission environment because several result tables are numerically dense.
- If the university template requires automatic cross-references for figures and tables, those will need to be added during final typesetting.
