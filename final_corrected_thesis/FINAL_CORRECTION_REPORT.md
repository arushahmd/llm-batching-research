## Final Correction Report

### Citation fixes made

- Restored in-text numbered citations in `chapter_2_literature_review_corrected.md` for the core background areas requested:
  Transformer architecture `[1]`, T5 `[2]`, Flan-T5 `[3]`, instruction following / InstructGPT `[4]`, LoRA `[5]`, Sentence-BERT `[6]`, FAISS `[7]`, curriculum learning `[8]`, ROUGE `[9]`, and BERTScore `[10]`.
- Added a `## References` section at the end of Chapter 2 using the supplied reference list.
- Verified that each reference number appears at least once in the chapter text.

### Figure order fixes made

- Reordered the Chapter 4 figure placeholders so they now appear in sequential textual order:
  Figure 4.1, Figure 4.2, Figure 4.3, Figure 4.4.
- Moved the Figure 4.2 placeholder upward so the semantic grouping pipeline now appears before the Figure 4.3 two-phase training flow placeholder.
- Preserved the Chapter 4 caveat that mixed batching was exploratory only and not part of the final production comparison.

### Figure 5.5 changes made

- Updated `scripts/reporting/generate_thesis_figures.py` so `figure_5_5_length_curriculum_extended_training.png` is generated as a dot plot rather than a horizontal bar plot.
- Split the visual into two clearly separated parts:
  same-budget comparison for Easy -> Hard and Hard -> Easy, plus a separate longer-training extension point for `exp_032`.
- Added the explicit on-figure note: `exp_032 uses longer training; not same-budget comparable`.
- Kept the underlying values unchanged:
  `11.338643074035645`, `11.337978998819986`, and `10.985996882120768`.

### Dense figure label changes made

- Reduced the on-plot numeric precision for Figure 5.4 and Figure 5.6 to compact 5-decimal labels.
- Regenerated Figure 1.1 with wider spacing and a less crowded final box.
- Rechecked Figure 5.7 and kept it label-light; the figure already relied on points and mean markers rather than dense numeric annotations.
- Left exact values unchanged in tables and chapter text.

### Remaining issues before Word/PDF formatting

- The thesis is still in Markdown with figure placeholders rather than final Word caption objects and automatic cross-references.
- Table wrapping, page breaks, and caption spacing still need to be handled in the final Word/PDF layout pass.
- Chapter 2 references are local to the chapter. If the final thesis uses a single end-of-thesis bibliography, these entries should be merged into that final reference section.
- If the department template requires a specific citation or bibliography style, the current numbered Markdown references will need to be mapped into that template.
