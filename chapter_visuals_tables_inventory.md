# 1. Files Inspected

No standalone `.log` files were found. Execution provenance is stored in `run_manifest.json`, `run_summary.json`, `phase_summary.json`, aggregated CSV/JSON reports, and generation-evaluation JSON/CSV files.

## Core files

| Path | Useful information |
|---|---|
| `D:\Working\llm-batching-research\README.md` | High-level study description, experiment blocks, and stated metrics. |
| `D:\Working\llm-batching-research\manifests\project_manifest.json` | Canonical local paths for base model, embedding model, dataset variants, processed datasets, and semantic indexes. |
| `D:\Working\llm-batching-research\requirements.txt` | Software package versions used by the local pipeline. |
| `D:\Working\llm-batching-research\metadata\experiment_registry.csv` | Inspected; empty file, so it is not usable as a registry table. |
| `D:\Working\llm-batching-research\reports.zip` | Zipped snapshot of report artifacts; listing confirms report and generation-eval packaging. |

## Configs inspected

| Path | Useful information |
|---|---|
| `D:\Working\llm-batching-research\configs\experiments\exp_012_random_only_multiseed.yaml` to `exp_015_random_to_grouped_multiseed.yaml` | 1k short experiment group, seeds `13,21,42`, epochs `0.2 + 0.2`. |
| `D:\Working\llm-batching-research\configs\experiments\exp_018_random_only_multiseed_long.yaml` to `exp_021_random_to_grouped_multiseed_long.yaml` | 1k long experiment group, epochs `0.5 + 0.5`. |
| `D:\Working\llm-batching-research\configs\experiments\exp_022_random_only_multiseed_3k.yaml` to `exp_025_random_to_grouped_multiseed_3k.yaml` | 3k experiment group. |
| `D:\Working\llm-batching-research\configs\experiments\exp_026_random_only_multiseed_5k.yaml` to `exp_029_random_to_grouped_multiseed_5k.yaml` | 5k main comparison group. |
| `D:\Working\llm-batching-research\configs\experiments\exp_030_easy_to_hard_length_multiseed_5k.yaml` | 5k easy-to-hard length curriculum. |
| `D:\Working\llm-batching-research\configs\experiments\exp_031_hard_to_easy_length_multiseed_5k.yaml` | 5k hard-to-easy length curriculum. |
| `D:\Working\llm-batching-research\configs\experiments\exp_032_hard_to_easy_length_longer_training_5k.yaml` | Extended hard-to-easy run with epochs `1.0 + 1.0`. |
| `D:\Working\llm-batching-research\configs\experiments\exp_016_random_to_grouped_high_refine_lr_multiseed.yaml` | Inspected; zero-byte placeholder. |
| `D:\Working\llm-batching-research\configs\experiments\exp_017_random_to_grouped_lr_control.yaml` | Inspected; zero-byte placeholder. |
| `D:\Working\llm-batching-research\configs\datasets\dolly_small_1k.yaml` | Inspected; zero-byte file. |
| `D:\Working\llm-batching-research\configs\models\flan_t5_small_lora.yaml` | Inspected; zero-byte file. |

## Scripts and source inspected

| Path | Useful information |
|---|---|
| `D:\Working\llm-batching-research\scripts\training\run_multiseed.py` | Multiseed launcher. |
| `D:\Working\llm-batching-research\scripts\training\run_experiment.py` | Main single-seed training pipeline; writes run manifests. |
| `D:\Working\llm-batching-research\src\training\experiment_runner.py` | Two-phase orchestration; phase-1 adapter carried into phase 2. |
| `D:\Working\llm-batching-research\src\training\phase_runner.py` | Per-phase train/eval wrapper. |
| `D:\Working\llm-batching-research\src\training\trainer_factory.py` | Exact `Seq2SeqTrainingArguments`, LoRA application, sampler injection. |
| `D:\Working\llm-batching-research\src\batching\samplers.py` | Production sampler modes: `random`, `grouped`, `easy_to_hard_length`, `hard_to_easy_length`. |
| `D:\Working\llm-batching-research\src\batching\curriculum_sampler.py` | Length curriculum implementation based on tokenized sequence length. |
| `D:\Working\llm-batching-research\src\batching\grouping.py` | Semantic grouping logic and anchor-to-group mapping. |
| `D:\Working\llm-batching-research\src\batching\index_loader.py` | Semantic-index bundle validation and loading. |
| `D:\Working\llm-batching-research\src\config\loader.py` | Experiment-config validation and path resolution from the manifest. |
| `D:\Working\llm-batching-research\src\data\raw_loader.py` | Raw schema expectations and embedding text construction. |
| `D:\Working\llm-batching-research\src\data\processed_loader.py` | Processed dataset split/column validation. |
| `D:\Working\llm-batching-research\src\data\alignment.py` | Raw-to-processed alignment used for semantic grouping. |
| `D:\Working\llm-batching-research\scripts\data\create_dolly_subset.py` | 3k raw subset creation (`seed=42`, `3000` rows). |
| `D:\Working\llm-batching-research\scripts\data\create_dolly_5k_subset.py` | 5k raw subset creation (`seed=42`, `5000` rows). |
| `D:\Working\llm-batching-research\scripts\data\process_dolly_3k.py` | 3k tokenization pipeline: `MAX_LENGTH=512`, `TRAIN_RATIO=0.9`, adds `raw_idx`. |
| `D:\Working\llm-batching-research\scripts\data\process_dolly_5k.py` | 5k tokenization pipeline with the same settings. |
| `D:\Working\llm-batching-research\scripts\indexing\build_embeddings_dolly_3k.py` | 3k semantic embeddings, normalized, `TEXT_MODE=instruction_plus_context`. |
| `D:\Working\llm-batching-research\scripts\indexing\build_neighbors_dolly_3k.py` | 3k neighbor graph with `TOP_K=8`. |
| `D:\Working\llm-batching-research\scripts\indexing\build_embeddings_dolly_5k.py` | 5k semantic embeddings, normalized, `TEXT_MODE=instruction_plus_context`. |
| `D:\Working\llm-batching-research\scripts\indexing\build_neighbors_dolly_5k.py` | 5k neighbor graph with `TOP_K=8`. |
| `D:\Working\llm-batching-research\scripts\evaluation\evaluate_generation_quality.py` | Deterministic generation-evaluation protocol (`batch_size=8`, `max_input_length=512`, `max_new_tokens=128`). |
| `D:\Working\llm-batching-research\scripts\reporting\aggregate_results.py` | Official optimization-metric aggregation into `master_summary_table.csv`. |
| `D:\Working\llm-batching-research\scripts\reporting\generate_plots.py` | Current plotting script. |
| `D:\Working\llm-batching-research\scripts\reporting\generate_tables.py` | Inspected; zero-byte file. |
| `D:\Working\llm-batching-research\scripts\maintenance\collect_research_bundle_local.py` | Export-bundle packager; confirms `exports\research_bundle_local` is a snapshot. |

## Notebooks inspected

| Path | Useful information |
|---|---|
| `D:\Working\llm-batching-research\notebooks\01_data\build_embeddings_faiss.ipynb` | Exploratory embedding/FAISS pipeline. |
| `D:\Working\llm-batching-research\notebooks\02_batching\grouped_minibatches.ipynb` | Exploratory grouped-mini-batch construction notebook. |
| `D:\Working\llm-batching-research\notebooks\03_training\train_random.ipynb` | Baseline random training notebook. |
| `D:\Working\llm-batching-research\notebooks\03_training\train_grouped.ipynb` | Grouped training notebook. |
| `D:\Working\llm-batching-research\notebooks\03_training\train_lora_two_phase_ablation.ipynb` | Two-phase curriculum notebook. |
| `D:\Working\llm-batching-research\notebooks\03_training\train_lora_sampler_ablation.ipynb` | Explicitly mentions mixed grouped + random batching; exploratory only. |
| `D:\Working\llm-batching-research\notebooks\04_evaluation\eval_metrics.ipynb` and `eval_metrics_dynamic.ipynb` | Evaluation dashboard notebooks. |

## Result and artifact files inspected

| Path | Useful information |
|---|---|
| `D:\Working\llm-batching-research\reports\master\master_summary_table.csv` | Current official aggregated optimization results across `exp_012` to `exp_032`. |
| `D:\Working\llm-batching-research\reports\master\master_per_seed_results.csv` | Current per-seed optimization metrics. |
| `D:\Working\llm-batching-research\reports\master\master_summary.json` | JSON version of the master summary. |
| `D:\Working\llm-batching-research\reports\archive\exp_012_to_exp_015_summary_table.csv` | Archived 1k short summary block. |
| `D:\Working\llm-batching-research\reports\archive\exp_018_to_exp_021_summary_table.csv` | Archived 1k long summary block; conflicts with current master for `exp_018`. |
| `D:\Working\llm-batching-research\reports\archive\exp_022_to_exp_025_summary_table.csv` | Archived 3k summary block. |
| `D:\Working\llm-batching-research\reports\archive\exp_012_to_exp_029_summary_table.csv` | Older cross-block aggregation for the four batching methods only. |
| `D:\Working\llm-batching-research\reports\generation_eval_summaries\generation_metrics_summary_table.csv` | Aggregated generation metrics for `exp_026` to `exp_029`. |
| `D:\Working\llm-batching-research\reports\generation_eval_5k\generation_metrics_summary_table.csv` | Top-level aggregated generation metrics for `exp_030` and `exp_031` only. |
| `D:\Working\llm-batching-research\reports\generation_eval_5k\exp_026_random_only_multiseed_5k\seed_013\generation_metrics.json` | Example per-seed generation JSON; overlaps with aggregated CSVs but does not match them numerically. |
| `D:\Working\llm-batching-research\reports\plots\combined_plot_data.csv` | Plot-data snapshot with only `16` rows; omits length-curriculum rows and `exp_032`. |
| `D:\Working\llm-batching-research\exports\research_bundle_local\README_bundle.txt` | Export bundle snapshot reporting only `8` plot files and `48` run summaries. |

# 2. Existing Figures / Images Found

| File Path | What It Shows | Suitable for Thesis? | Recommended Chapter/Section | Needs Redesign? | Notes |
|---|---|---|---|---|---|
| `D:\Working\llm-batching-research\reports\plots\plot_1k_short_eval_loss.png`<br>`D:\Working\llm-batching-research\mnt\data\plot_1k_short_eval_loss.png`<br>`D:\Working\llm-batching-research\mnt\plots\plot_1k_short_eval_loss.png`<br>`D:\Working\llm-batching-research\exports\research_bundle_local\plots\plot_1k_short_eval_loss.png` | Mean eval loss for the four 1k short methods. | REDESIGN | Chapter 5, 1k short results | Yes | Current styling is generic and the differences are visually tiny. |
| `D:\Working\llm-batching-research\reports\plots\plot_1k_short_phase2_delta.png`<br>`D:\Working\llm-batching-research\mnt\data\plot_1k_short_phase2_delta.png`<br>`D:\Working\llm-batching-research\mnt\plots\plot_1k_short_phase2_delta.png`<br>`D:\Working\llm-batching-research\exports\research_bundle_local\plots\plot_1k_short_phase2_delta.png` | Mean phase-2 eval-loss delta for the four 1k short methods. | REDESIGN | Chapter 5, phase-wise analysis | Yes | Use a cleaner thesis figure with uncertainty or direct value labels. |
| `D:\Working\llm-batching-research\reports\plots\plot_1k_long_eval_loss.png`<br>`D:\Working\llm-batching-research\mnt\data\plot_1k_long_eval_loss.png`<br>`D:\Working\llm-batching-research\mnt\plots\plot_1k_long_eval_loss.png`<br>`D:\Working\llm-batching-research\exports\research_bundle_local\plots\plot_1k_long_eval_loss.png` | Mean eval loss for the four 1k long methods. | REDESIGN | Chapter 5, 1k long results | Yes | Data is usable; rendering is not thesis-grade. |
| `D:\Working\llm-batching-research\reports\plots\plot_1k_long_phase2_delta.png`<br>`D:\Working\llm-batching-research\mnt\data\plot_1k_long_phase2_delta.png`<br>`D:\Working\llm-batching-research\mnt\plots\plot_1k_long_phase2_delta.png`<br>`D:\Working\llm-batching-research\exports\research_bundle_local\plots\plot_1k_long_phase2_delta.png` | Mean phase-2 eval-loss delta for the four 1k long methods. | REDESIGN | Chapter 5, phase-wise analysis | Yes | Same caveat as the 1k short delta figure. |
| `D:\Working\llm-batching-research\reports\plots\plot_3k_long_eval_loss.png`<br>`D:\Working\llm-batching-research\mnt\data\plot_3k_long_eval_loss.png`<br>`D:\Working\llm-batching-research\mnt\plots\plot_3k_long_eval_loss.png`<br>`D:\Working\llm-batching-research\exports\research_bundle_local\plots\plot_3k_long_eval_loss.png` | Mean eval loss for the four 3k methods. | REDESIGN | Chapter 5, 3k results | Yes | Duplicate copies are older snapshots. |
| `D:\Working\llm-batching-research\reports\plots\plot_3k_long_phase2_delta.png`<br>`D:\Working\llm-batching-research\mnt\data\plot_3k_long_phase2_delta.png`<br>`D:\Working\llm-batching-research\mnt\plots\plot_3k_long_phase2_delta.png`<br>`D:\Working\llm-batching-research\exports\research_bundle_local\plots\plot_3k_long_phase2_delta.png` | Mean phase-2 eval-loss delta for the four 3k methods. | REDESIGN | Chapter 5, phase-wise analysis | Yes | Regenerate with clearer annotations. |
| `D:\Working\llm-batching-research\reports\plots\plot_5k_long_eval_loss.png` | Mean eval loss for the four 5k long methods. | REDESIGN | Chapter 5, 5k main comparison | Yes | Usable source data, but not publication-grade styling. |
| `D:\Working\llm-batching-research\reports\plots\plot_5k_long_phase2_delta.png` | Mean phase-2 eval-loss delta for the four 5k long methods. | REDESIGN | Chapter 5, 5k phase-wise comparison | Yes | Useful metric, but current rendering is generic. |
| `D:\Working\llm-batching-research\reports\plots\plot_5k_full_curriculum_comparison_eval_loss.png` | Intended 5k batching-vs-curriculum comparison. | IGNORE | NOT AVAILABLE | Yes | Misleading current file: the title implies length-curriculum inclusion, but the rendered bars omit `exp_030` and `exp_031`. |
| `D:\Working\llm-batching-research\reports\plots\plot_random_baseline_cross_block_eval_loss.png` | Random-baseline eval loss across `1k_short`, `1k_long`, `3k_long`, `5k_long`. | APPENDIX ONLY | Appendix, baseline diagnostic | Optional | Useful as a diagnostic, but too narrow for the main thesis. |
| `D:\Working\llm-batching-research\reports\plots\plot_cross_block_eval_trends.png`<br>`D:\Working\llm-batching-research\mnt\data\plot_cross_block_eval_trends.png`<br>`D:\Working\llm-batching-research\mnt\plots\plot_cross_block_eval_trends.png`<br>`D:\Working\llm-batching-research\exports\research_bundle_local\plots\plot_cross_block_eval_trends.png` | Older multi-method cross-block line plot. | IGNORE | NOT AVAILABLE | Yes | Stale and incomplete: it only reaches `3k_long` and is tied to incomplete plot-data snapshots. |

# 3. Tables Suitable for Chapter 4: Experimental Setup

## Overview

| Table Title | Source File(s) | Recommended Section | Columns | Complete? | Caveats |
|---|---|---|---|---|---|
| Dataset Splits | Run manifests for `exp_012`, `exp_022`, `exp_026`; subset scripts | 4.1 Data | Dataset, raw rows, processed train rows, processed eval rows, split ratio, seed | Partial | 1k raw-subset creation is not directly scripted in the inspected repo. |
| Dataset Preprocessing and Tokenization Settings | `process_dolly_3k.py`, `process_dolly_5k.py`, `raw_loader.py`, `processed_loader.py` | 4.1 Data | Input formatting, target formatting, max length, padding, truncation, split strategy, saved columns | Partial | No dedicated `process_dolly_1k.py` found. |
| Base Model Setup | `project_manifest.json`, `models\flan-t5-small\config.json`, run manifests | 4.2 Model | Model ID, architecture, layers, heads, hidden size, max positions, vocab size | Yes | Model-config YAML is empty, so cite manifest/model JSON instead. |
| LoRA Configuration | Experiment YAMLs, `adapter_config.json` | 4.2 Model Adaptation | `r`, `alpha`, dropout, target modules, bias, task type | Yes | Use the saved adapter config as implementation confirmation. |
| Training Configuration | Experiment YAMLs, `trainer_factory.py` | 4.3 Training | Batch sizes, grad accumulation, eval frequency, save strategy, precision flags, learning rates, phase epochs | Yes | Phase epochs vary by block and must be shown. |
| Semantic Grouping Configuration | Grouping/indexing scripts, grouped run manifests | 4.4 Semantic Grouping | Embedding model, text mode, normalization, top-k, max group size, include anchor | Partial | 1k index differs materially from 3k/5k. |
| Batch Scheduling Strategies | Experiment YAMLs, `samplers.py`, `curriculum_sampler.py` | 4.5 Scheduling Strategies | Strategy, phase 1, phase 2, semantic index requirement, ordering rule | Yes | Mixed grouped + random exists only in exploratory notebooks. |
| Experiment Groups | Experiment YAMLs, run manifests | 4.6 Experiment Matrix | Block, dataset, experiment IDs, methods, phase epochs | Yes | Exclude empty `exp_016` and `exp_017`. |
| Generation Evaluation Configuration | `evaluate_generation_quality.py`, generation summary folders | 4.7 Evaluation | Eval split reconstruction, batch size, max input length, max new tokens, decoding mode, metrics | Partial | Only 5k generation outputs are present; none for `exp_032`. |
| Hardware and Software Environment | Run manifests, `requirements.txt` | 4.8 Environment | Python, PyTorch, Transformers, Datasets, PEFT, CUDA availability, hardware | Partial | GPU model, VRAM, CPU, RAM, OS version are not persisted. |
| Known Uncertainties and Caveats | Empty configs, inconsistent generation summaries, stale plot data | 4.9 Limitations | Item, evidence, impact | Yes | These caveats should be stated explicitly in Chapter 4 or an appendix note. |

## Table Content

### 3.1 Dataset Splits

| Dataset Variant | Raw Dataset Path Used by Runs | Raw Source Rows | Processed Train Rows | Processed Eval Rows | Split Ratio | Split Seed | Evidence |
|---|---|---:|---:|---:|---|---:|---|
| `dolly_small_1k` | `D:\Working\llm-batching-research\data\raw\dolly_15k` | 15011 | 900 | 100 | 90/10 | 42 | `exp_012`/`exp_013` run manifests |
| `dolly_3k` | `D:\Working\llm-batching-research\data\raw\dolly_3k` | 3000 | 2700 | 300 | 90/10 | 42 | `exp_022`/`exp_023` run manifests; `create_dolly_subset.py` |
| `dolly_5k` | `D:\Working\llm-batching-research\data\raw\dolly_5k` | 5000 | 4500 | 500 | 90/10 | 42 | `exp_026`/`exp_027`/`exp_030` run manifests; `create_dolly_5k_subset.py` |

### 3.2 Dataset Preprocessing and Tokenization Settings

| Setting | Value | Source | Caveat |
|---|---|---|---|
| Input text construction | `instruction` if no context; otherwise `instruction + "\n\nContext: " + context` | `process_dolly_3k.py`, `process_dolly_5k.py`, `raw_loader.py` | No dedicated 1k processing script found. |
| Target text | `response` | `process_dolly_3k.py`, `process_dolly_5k.py` | None |
| Tokenization | `truncation=True`, `padding="max_length"`, `max_length=512` for inputs and labels | `process_dolly_3k.py`, `process_dolly_5k.py` | 1k is inferred from artifacts, not a found script. |
| Split strategy | `train_test_split(test_size=0.1, seed=42)` | `process_dolly_3k.py`, `process_dolly_5k.py` | 1k split is inferred from run manifests. |
| Saved processed columns | `input_ids`, `attention_mask`, `labels`, `raw_idx` | `process_dolly_3k.py`, `process_dolly_5k.py`, `processed_loader.py` | None |

### 3.3 Base Model Setup

| Field | Value | Source |
|---|---|---|
| Base model ID | `google/flan-t5-small` | `project_manifest.json`, run manifests |
| Local path | `D:\Working\llm-batching-research\models\flan-t5-small` | `project_manifest.json` |
| Architecture | `T5ForConditionalGeneration` | `models\flan-t5-small\config.json` |
| Hidden size (`d_model`) | `512` | `models\flan-t5-small\config.json` |
| Feed-forward size (`d_ff`) | `1024` | `models\flan-t5-small\config.json` |
| Encoder layers | `8` | `models\flan-t5-small\config.json` |
| Decoder layers | `8` | `models\flan-t5-small\config.json` |
| Attention heads | `6` | `models\flan-t5-small\config.json` |
| Max positions | `512` | `models\flan-t5-small\config.json` |
| Vocab size | `32128` | `models\flan-t5-small\config.json` |
| Dropout rate | `0.1` | `models\flan-t5-small\config.json` |

### 3.4 LoRA Configuration

| Field | Value | Source |
|---|---|---|
| PEFT method | LoRA | Experiment YAMLs, `adapter_config.json` |
| Task type | `SEQ_2_SEQ_LM` | `adapter_config.json` |
| Rank `r` | `16` | Experiment YAMLs, `adapter_config.json` |
| LoRA alpha | `32` | Experiment YAMLs, `adapter_config.json` |
| LoRA dropout | `0.05` | Experiment YAMLs, `adapter_config.json` |
| Target modules | `q`, `v` | Experiment YAMLs, `adapter_config.json` |
| Bias | `none` | `trainer_factory.py`, `adapter_config.json` |

### 3.5 Training Configuration

| Setting | Value | Source | Caveat |
|---|---|---|---|
| Per-device train batch size | `8` | Experiment YAMLs, run manifests | None |
| Per-device eval batch size | `8` | Experiment YAMLs, run manifests | None |
| Gradient accumulation steps | `4` | Experiment YAMLs, run manifests | Effective batch size is not directly logged beyond this. |
| Weight decay | `0.01` | Experiment YAMLs | None |
| Eval strategy / steps | `steps`, `50` | Experiment YAMLs, `trainer_factory.py` | None |
| Save strategy / total limit | `epoch`, `2` | Experiment YAMLs, `trainer_factory.py` | None |
| Precision flags | `fp16=false`, `bf16=false` | Experiment YAMLs, run manifests | None |
| Learning rates | Phase 1 `7e-05`, Phase 2 `5e-05` | Experiment YAMLs | None |
| Phase epochs, 1k short | `0.2 + 0.2` | `exp_012` to `exp_015` | None |
| Phase epochs, 1k long / 3k / 5k / length curricula | `0.5 + 0.5` | `exp_018` to `exp_031` | None |
| Phase epochs, extended hard-to-easy | `1.0 + 1.0` | `exp_032` | None |

### 3.6 Semantic Grouping Configuration

| Field | 1k | 3k | 5k | Source | Caveat |
|---|---|---|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` | Manifest and index metadata | None |
| Semantic text mode | `instruction` only (older metadata schema) | `instruction_plus_context` | `instruction_plus_context` | 1k grouped run manifest; 3k/5k `meta.json` | Cross-scale inconsistency should be stated. |
| Effective neighbor width | `32` | `8` | `8` | grouped run manifests | The 1k index is materially different. |
| Group size in training | `max_group_size=8` | `8` | `8` | experiment configs, `grouping.py` | None |
| Include anchor | `true` | `true` | `true` | experiment configs, `grouping.py` | None |
| Similarity index | FAISS `IndexFlatIP` over normalized embeddings | same | same | indexing scripts | 1k build script was not found in `scripts\indexing`. |

# 4. Figures Suitable for Chapter 4: Experimental Setup

Methodology/setup figures should be newly generated from source code, manifests, and experiment configs. No inspected PNG file is a setup/methodology figure.

| Figure Title | Recommended Section | Source Content/Data | What It Should Show | Can Be Generated Now? | Caveats |
|---|---|---|---|---|---|
| Overall Experimental Pipeline | 4.1 Overview | `project_manifest.json`, dataset scripts, training scripts, reporting scripts | Dolly dataset variant selection -> tokenization -> optional semantic index -> two-phase LoRA fine-tuning -> aggregation -> generation evaluation | Yes | Use a schematic, not a result plot. |
| Semantic Grouping Pipeline | 4.4 Semantic Grouping | indexing scripts, `grouping.py`, `index_loader.py`, `alignment.py` | Input text -> embeddings -> FAISS search -> neighbor graph -> anchor-to-group map -> grouped batches | Yes | Annotate that 1k uses older index metadata and effective `top_k=32`. |
| Two-Phase Curriculum Training Flow | 4.5 Scheduling Strategies | experiment YAMLs, `experiment_runner.py`, `phase_runner.py` | Phase 1 sampler mode and LR -> adapter handoff -> Phase 2 sampler mode and LR | Yes | Show that the phase-1 adapter is reused, not restarted. |
| Length-Based Curriculum Pipeline | 4.5 Scheduling Strategies | `curriculum_sampler.py`, `exp_030` to `exp_032` configs | Processed dataset -> sequence-length scoring -> ascending/descending ordering -> batched training | Yes | "Easy" and "hard" are sequence-length heuristics, not semantic difficulty labels. |
| Dataset Split Diagram | 4.1 Data | run manifests for 1k/3k/5k | Raw-source rows and resulting processed train/eval splits by scale | Yes | For 1k, note that the raw path is the full 15k dataset while the processed split is 900/100. |
| Batch Scheduling Strategy Taxonomy | 4.5 Scheduling Strategies | experiment YAMLs, `samplers.py` | The six implemented strategies: random, grouped, grouped->random, random->grouped, easy->hard, hard->easy | Yes | Mixed batching appears only in exploratory notebooks and is not in production code. |

## Recommended Chapter 4 Captions

- **Overall Experimental Pipeline.** End-to-end workflow for instruction fine-tuning with two-phase batch scheduling, from Dolly subset preparation and tokenization through LoRA training, aggregation, and generation evaluation.
- **Semantic Grouping Pipeline.** Construction of grouped mini-batches using sentence embeddings, FAISS nearest-neighbor retrieval, raw-to-processed index alignment, and anchor-centered group assembly.
- **Two-Phase Curriculum Training Flow.** Two-phase fine-tuning procedure in which the sampler mode can remain fixed or change between phases while the phase-1 adapter is reused in phase 2.
- **Length-Based Curriculum Pipeline.** Length-driven curriculum sampler that orders examples by tokenized sequence length in ascending or descending order before batch formation.
- **Dataset Split Diagram.** Relationship between raw dataset scale and processed 90/10 train-eval split used in each experiment block.
- **Batch Scheduling Strategy Taxonomy.** Taxonomy of the six production-implemented scheduling strategies evaluated in the thesis.

# 5. Tables Suitable for Chapter 5: Results and Analysis

## Overview

| Table Title | Source File(s) | Experiment IDs Included | Columns | Thesis-Ready? | Caveats |
|---|---|---|---|---|---|
| 1k Short Training Comparison | `reports\master\master_summary_table.csv` | `exp_012` to `exp_015` | Method, exp_ids, N, Seeds, Mean Eval Loss, Std Eval Loss, Mean Train Loss, Mean Gap, Mean Phase2 Delta Eval | Yes | Differences are extremely small; interpret with care. |
| 1k Long Training Comparison | `reports\master\master_summary_table.csv` | `exp_018` to `exp_021` | same as above | Yes | Archived 1k-long summary file conflicts with current master for `exp_018`; use the current master table. |
| 3k Comparison | `reports\master\master_summary_table.csv` | `exp_022` to `exp_025` | same as above | Yes | None beyond normal small-effect caution. |
| 5k Main Comparison | `reports\master\master_summary_table.csv` | `exp_026` to `exp_029` | same as above | Yes | Strongest core comparison block. |
| 5k Length Curriculum Comparison | `reports\master\master_summary_table.csv` | `exp_030`, `exp_031` | same as above | Yes | Suitable as its own table or a subsection of the 5k results table. |
| Extended Hard-to-Easy Result | `reports\master\master_summary_table.csv` | `exp_031`, `exp_032` | Method, exp_ids, N, Seeds, Mean Eval Loss, Std Eval Loss, Mean Train Loss, Mean Gap, Mean Phase2 Delta Eval | Yes | `exp_032` is a longer-training extension, not a same-budget comparison. |
| Generation Metrics for 5k Semantic/Curriculum Experiments | `reports\generation_eval_summaries\generation_metrics_summary_table.csv`; `reports\generation_eval_5k\generation_metrics_summary_table.csv` | `exp_026` to `exp_031` | exp_id, n_seeds, seeds, mean/std ROUGE and BERTScore | Conditional | Aggregated generation CSVs conflict with overlapping per-seed `generation_metrics.json` files; cite the exact source file used. |
| Per-Seed Results Table | `reports\master\master_per_seed_results.csv` | Any chosen subset; 5k-focused subset is most useful | block, exp_id, method, seed, final_eval_loss, final_train_loss, generalization_gap, phase_2_delta_eval | Appendix Only | Too verbose for main Chapter 5. |
| Variance / Std Table | `reports\master\master_summary_table.csv`; generation summary tables | block/method or exp_id-level std values | Std Eval Loss, Std Train Loss, Std Gap, Std Phase2 Delta Eval, generation std metrics | Appendix Only | Main tables already contain the std columns. |

## Table Content

### 5.1 1k Short Training Comparison

| Method | exp_ids | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Random | exp_012_random_only_multiseed | 3 | 13,21,42 | 9.616399129231771 | 0.00012846433173675242 | 38.770100063747826 | -29.153700934516056 | -0.0013834635416666667 |
| Grouped | exp_013_grouped_only_multiseed | 3 | 13,21,42 | 9.61652692159017 | 0.00021044965469782815 | 38.722209506564674 | -29.105682584974502 | -0.0013230641682942708 |
| Grouped->Random | exp_014_grouped_to_random_multiseed | 3 | 13,21,42 | 9.616591453552246 | 1.2615925364802315e-05 | 38.77133009168837 | -29.15473863813612 | -0.001311937967936198 |
| Random->Grouped | exp_015_random_to_grouped_multiseed | 3 | 13,21,42 | 9.616386731465658 | 9.03947098320946e-05 | 38.722546895345054 | -29.106160163879395 | -0.0013634363810221355 |

### 5.2 1k Long Training Comparison

| Method | exp_ids | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Random | exp_018_random_only_multiseed_long | 3 | 13,21,42 | 9.611108779907227 | 0.00018671569539907427 | 38.760330539279515 | -29.149221759372285 | -0.004041353861490886 |
| Grouped | exp_019_grouped_only_multiseed_long | 3 | 13,21,42 | 9.611552556355795 | 0.00010812468756713155 | 38.722173394097226 | -29.11062083774143 | -0.0038159688313802085 |
| Grouped->Random | exp_020_grouped_to_random_multiseed_long | 3 | 13,21,42 | 9.611759503682455 | 0.00021096115141597821 | 38.76272142198351 | -29.150961918301054 | -0.0037781397501627603 |
| Random->Grouped | exp_021_random_to_grouped_multiseed_long | 3 | 13,21,42 | 9.611469268798828 | 0.0002447879672933447 | 38.722423977322045 | -29.11095470852322 | -0.003803253173828125 |

### 5.3 3k Comparison

| Method | exp_ids | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Random | exp_022_random_only_multiseed_3k | 3 | 13,21,42 | 11.423059145609537 | 0.0022963785111432176 | 45.43341644050539 | -34.010357294895854 | -0.03380934397379557 |
| Grouped | exp_023_grouped_only_multiseed_3k | 3 | 13,21,42 | 11.425052642822266 | 0.0006824924120630219 | 45.453193368837816 | -34.02814072601555 | -0.032903035481770836 |
| Grouped->Random | exp_024_grouped_to_random_multiseed_3k | 3 | 13,21,42 | 11.424866358439127 | 0.001265824868793103 | 45.442404103833574 | -34.01753774539445 | -0.03327242533365885 |
| Random->Grouped | exp_025_random_to_grouped_multiseed_3k | 3 | 13,21,42 | 11.425841649373373 | 0.000684287960140967 | 45.455506317375246 | -34.02966466800187 | -0.032347679138183594 |

### 5.4 5k Main Comparison

| Method | exp_ids | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Random | exp_026_random_only_multiseed_5k | 3 | 13,21,42 | 11.341803868611654 | 0.004030287838901957 | 45.12238576826355 | -33.7805818996519 | -0.07824929555257161 |
| Grouped | exp_027_grouped_only_multiseed_5k | 3 | 13,21,42 | 11.346779187520346 | 0.0025083908611327086 | 45.171566063249614 | -33.82478687572927 | -0.07535489400227864 |
| Grouped->Random | exp_028_grouped_to_random_multiseed_5k | 3 | 13,21,42 | 11.344831148783365 | 0.0009995714022504269 | 45.13616887840307 | -33.7913377296197 | -0.07722759246826172 |
| Random->Grouped | exp_029_random_to_grouped_multiseed_5k | 3 | 13,21,42 | 11.343379974365234 | 0.002895475059662966 | 45.15890305926543 | -33.81552308490019 | -0.07583808898925781 |

### 5.5 5k Length Curriculum Comparison

| Method | exp_ids | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Easy->Hard Length | exp_030_easy_to_hard_length_multiseed_5k | 3 | 13,21,42 | 11.338643074035645 | 0.0012300206513966253 | 45.13238579118755 | -33.79374271715191 | -0.0795135498046875 |
| Hard->Easy Length | exp_031_hard_to_easy_length_multiseed_5k | 3 | 13,21,42 | 11.337978998819986 | 0.0009772446107854716 | 45.12975311279297 | -33.791774113972984 | -0.08008988698323567 |

### 5.6 Extended Hard-to-Easy Result

| Method | exp_ids | N | Seeds | Mean Eval Loss | Std Eval Loss | Mean Train Loss | Mean Gap | Mean Phase2 Delta Eval |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Hard->Easy Length | exp_031_hard_to_easy_length_multiseed_5k | 3 | 13,21,42 | 11.337978998819986 | 0.0009772446107854716 | 45.12975311279297 | -33.791774113972984 | -0.08008988698323567 |
| Hard->Easy Length (Longer Training) | exp_032_hard_to_easy_length_longer_training_5k | 3 | 13,21,42 | 10.985996882120768 | 0.008129767881074827 | 44.02101001784593 | -33.03501313572516 | -0.2848030726114909 |

### 5.7 Generation Metrics for 5k Semantic/Curriculum Experiments

Source-file caveat: the batching-method generation summary and the length-curriculum generation summary are split across two top-level folders, and both aggregated CSVs conflict with overlapping per-seed `generation_metrics.json` files. The table below preserves the exact values from the two aggregated summary CSVs only.

| exp_id | n_seeds | seeds | mean_rouge1 | std_rouge1 | mean_rouge2 | std_rouge2 | mean_rougeL | std_rougeL | mean_rougeLsum | std_rougeLsum | mean_bertscore_f1 | std_bertscore_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp_026_random_only_multiseed_5k | 3 | seed_013,seed_021,seed_042 | 0.1332991221165225 | 0.0020164915902323 | 0.05172597342614166 | 0.001973474178185322 | 0.12528074807592213 | 0.001879402357167189 | 0.1255176984312918 | 0.001713859683307618 | 0.8358021107912063 | 0.0004622980477068232 |
| exp_027_grouped_only_multiseed_5k | 3 | seed_013,seed_021,seed_042 | 0.13515864447757928 | 0.000821310043745659 | 0.05267632000153163 | 0.0008599748859510768 | 0.12741708266264154 | 0.0008997847037711997 | 0.12762222208271706 | 0.0009568022221241892 | 0.8365236309369406 | 0.00029145341957466175 |
| exp_028_grouped_to_random_multiseed_5k | 3 | seed_013,seed_021,seed_042 | 0.13573552053686477 | 0.0006310815717993919 | 0.05290762532282947 | 0.0007638326220628706 | 0.12741556756984396 | 0.00040348389836700736 | 0.1279101682552873 | 0.00047559550741240033 | 0.836249907930692 | 0.00025796425857546053 |
| exp_029_random_to_grouped_multiseed_5k | 3 | seed_013,seed_021,seed_042 | 0.13429778290121783 | 0.0009638989478806322 | 0.05139917328310489 | 0.0006283945874624924 | 0.12623164340206786 | 0.0008984621429260791 | 0.1263183377758188 | 0.0008879813126434798 | 0.8363362034161885 | 0.0004125665273872965 |
| exp_030_easy_to_hard_length_multiseed_5k | 3 | seed_013,seed_021,seed_042 | 0.1349535952911857 | 0.0022750204251505116 | 0.051761622149622204 | 0.0019208392324216834 | 0.12602883537238033 | 0.0020777819486155744 | 0.12640203541955244 | 0.002185766668899262 | 0.8361860071023305 | 0.00028109593218276314 |
| exp_031_hard_to_easy_length_multiseed_5k | 3 | seed_013,seed_021,seed_042 | 0.13569431685742261 | 0.001371932873660898 | 0.052930931439159314 | 0.0014885726613267686 | 0.12673431381120112 | 0.0014466042675307454 | 0.12715204647000258 | 0.0011848330035887271 | 0.8362131182750067 | 0.00021941865358977844 |

### 5.8 Per-Seed and Variance Tables

Per-seed optimization tables are available in `reports\master\master_per_seed_results.csv`. Variance and standard-deviation values are already present in `master_summary_table.csv` and the generation summary CSVs, but they are better used in the appendix than in the main chapter.

# 6. Figures Suitable for Chapter 5: Results and Analysis

| Figure Title | Source File(s) | X-axis | Y-axis | Grouping Variable | Recommended Chapter/Section | Caption | Caveats |
|---|---|---|---|---|---|---|---|
| Mean Eval Loss Comparison Across Methods | `reports\master\master_summary_table.csv`; existing `reports\plots\plot_1k_short_eval_loss.png`, `plot_1k_long_eval_loss.png`, `plot_3k_long_eval_loss.png`, `plot_5k_long_eval_loss.png` | Method | Mean Eval Loss | Experiment block | 5.1 Optimization Results | Mean final evaluation loss across batching methods within each experiment block. | Existing plots exist but should be redesigned as a multi-panel thesis figure. |
| Eval Loss by Dataset Scale | `reports\master\master_summary_table.csv`; existing `plot_random_baseline_cross_block_eval_loss.png`; stale `plot_cross_block_eval_trends.png` | Dataset scale/block | Mean Eval Loss | Method | 5.1 Cross-Scale Comparison | Mean evaluation loss as dataset scale increases from 1k to 5k. | Must be generated or heavily redesigned; the existing multi-method cross-block plot is stale and incomplete. |
| Random vs Grouped Comparison | `reports\master\master_summary_table.csv` | Block | Mean Eval Loss or Eval Loss Difference | Method pair (`Random`, `Grouped`) | 5.2 Random vs Semantic Grouping | Direct comparison of random batching and grouped batching across the 1k, 3k, and 5k result blocks. | Must be generated. |
| Curriculum Ordering Comparison: Random->Grouped vs Grouped->Random | `reports\master\master_summary_table.csv` | Block | Mean Eval Loss or Phase2 Delta Eval | Curriculum direction | 5.3 Two-Phase Curriculum Ordering | Comparison of the two two-phase semantic curricula under matched budgets. | Must be generated. |
| Length Curriculum Comparison: Easy->Hard vs Hard->Easy | `reports\master\master_summary_table.csv` | Method | Mean Eval Loss / Mean Phase2 Delta Eval | Length curriculum direction | 5.4 Length Curriculum Results | Comparison of ascending-length and descending-length curricula on Dolly 5k. | Must be generated; the current `plot_5k_full_curriculum_comparison_eval_loss.png` is not usable. |
| Extended Hard->Easy Comparison | `reports\master\master_summary_table.csv` | Training budget / experiment ID | Mean Eval Loss, Mean Phase2 Delta Eval | Hard->Easy configuration | 5.4 Length Curriculum Results | Comparison of standard-budget and longer-budget hard-to-easy length curricula. | Must be generated; `exp_032` has no existing figure. |
| Generation Metrics Comparison | `reports\generation_eval_summaries\generation_metrics_summary_table.csv`; `reports\generation_eval_5k\generation_metrics_summary_table.csv` | Experiment ID or strategy | ROUGE / BERTScore | Metric type | 5.5 Generation Quality | Generation-quality comparison across 5k batching and curriculum experiments. | Must be generated; aggregated CSVs conflict with overlapping per-seed JSONs. |
| Seed Variance / Standard Deviation Plot | `reports\master\master_summary_table.csv`; `reports\master\master_per_seed_results.csv` | Method or experiment block | Std Eval Loss or per-seed Final Eval Loss | Method / seed | 5.6 Stability and Variance | Variability of optimization results across seeds. | Must be generated. |
| Phase 2 Delta Eval Comparison | `reports\master\master_summary_table.csv`; existing phase2-delta plots for 1k/3k/5k | Method | Mean Phase2 Delta Eval | Experiment block | 5.3 Phase-Wise Analysis | Phase-2 contribution to evaluation-loss change under each scheduling strategy. | Existing plots exist but should be redesigned. |

# 7. Appendix-Only Material

| Source File Path | Why It Belongs in Appendix | Suggested Appendix Title |
|---|---|---|
| `D:\Working\llm-batching-research\reports\master\master_per_seed_results.csv` | Full per-seed optimization table is too verbose for the main results chapter. | Appendix A: Per-Seed Optimization Results |
| `D:\Working\llm-batching-research\reports\generation_eval_summaries\generation_metrics_per_seed.csv` | Useful for reproducibility and seed-level generation variance, but too detailed for the main chapter. | Appendix B: Per-Seed Generation Metrics for 5k Batching Runs |
| `D:\Working\llm-batching-research\reports\generation_eval_5k\generation_metrics_per_seed.csv` | Same as above for the length-curriculum subset. | Appendix C: Per-Seed Generation Metrics for 5k Length Curricula |
| `D:\Working\llm-batching-research\reports\generation_eval_5k\exp_0xx...\seed_xxx\predictions.csv` | Raw prediction dumps are useful for qualitative audit trails, not for main Chapter 5 tables. | Appendix D: Sample Generations and Prediction Dumps |
| `D:\Working\llm-batching-research\reports\archive\*.csv` and `*.json` | Archived block-wise summaries are useful for provenance, but the current master summary is the cleaner main-chapter source. | Appendix E: Archived Aggregation Snapshots |
| `D:\Working\llm-batching-research\experiments\exp_0xx...\seed_xxx\run_manifest.json` | Detailed execution provenance and environment snapshots are too granular for the main chapter. | Appendix F: Run Manifests and Environment Snapshots |
| `D:\Working\llm-batching-research\experiments\exp_0xx...\seed_xxx\run_summary.json` and `phase_summary.json` | Full phase-level logs are useful for auditability and replication, not for the main narrative. | Appendix G: Phase-Level Training Summaries |
| `D:\Working\llm-batching-research\reports\plots\plot_random_baseline_cross_block_eval_loss.png` | Useful diagnostic, but it only covers one method and is weaker than a regenerated full cross-scale figure. | Appendix H: Baseline Diagnostic Plots |
| `D:\Working\llm-batching-research\mnt\*.png` and `D:\Working\llm-batching-research\exports\research_bundle_local\plots\*.png` | Duplicate or stale plot copies should not be used in the main thesis, but can be retained as archival snapshots. | Appendix I: Archived Plot Snapshots |
| `D:\Working\llm-batching-research\reports.zip` | Packaged report archive is useful for replication packaging, not for the thesis body. | Appendix J: Supplementary Artifact Package |

# 8. Missing or Weak Visuals

| Missing Visual/Table | Why Useful | Missing Data Needed | Possible Alternative |
|---|---|---|---|
| Fully documented 1k raw-subset construction table | Would make the 1k experimental setup reproducible at the same provenance level as 3k and 5k. | A dedicated 1k subset creation script or manifest showing how `dolly_small_1k` was derived from Dolly 15k. | State the processed 900/100 split from run manifests and explicitly note the missing raw-subset script. |
| Hardware specification table with GPU model and memory | Important for a production-grade experimental setup chapter. | Saved GPU name, VRAM, CPU, RAM, and OS details. | Use the software-environment table and mark hardware rows as `NOT AVAILABLE`. |
| Training-dynamics line plots across steps/epochs | Would support deeper Chapter 5 analysis of convergence behavior beyond endpoint summaries. | Saved step-wise training/eval history logs or CSV exports from Trainer. | Use endpoint result tables plus phase-2 delta analysis. |
| Up-to-date combined plot data including length curricula and `exp_032` | Necessary for a fully current plot suite. | Regenerated `reports\plots\combined_plot_data.csv` from the current master summary. | Generate new thesis plots directly from `reports\master\master_summary_table.csv`. |
| Consistent generation-evaluation master table for `exp_026` to `exp_031` | A single canonical table would avoid source-folder splitting and conflicting aggregates. | One regenerated generation summary built from the chosen canonical per-seed source. | Combine the two existing summary CSVs, and state the conflict with per-seed JSON files. |
| Generation evaluation for `exp_032` | Would complete the longer-training length-curriculum analysis. | Per-seed `generation_metrics.json` and aggregated CSV/JSON outputs for `exp_032`. | Restrict generation-quality discussion to `exp_026` to `exp_031`. |
| Current multi-method cross-block plot | Useful for showing block-level behavior across all strategies. | A regenerated figure from the current master summary. | Replace the stale `plot_cross_block_eval_trends.png` with a new figure. |

# 9. Final Recommendation

## Chapter 4 should include:

- Tables:
- Dataset splits
- Base model and LoRA configuration
- Training configuration
- Semantic grouping configuration
- Batch scheduling strategies and experiment groups
- Generation evaluation configuration
- Figures:
- Overall experimental pipeline
- Semantic grouping pipeline
- Two-phase curriculum training flow
- Batch scheduling strategy taxonomy

## Chapter 5 should include:

- Tables:
- 1k short training comparison
- 1k long training comparison
- 3k comparison
- 5k main comparison
- 5k length curriculum comparison
- Extended hard-to-easy result
- Combined 5k generation metrics table for `exp_026` to `exp_031`, with an explicit source-file caveat
- Figures:
- A regenerated multi-panel mean-eval-loss figure for 1k/3k/5k blocks
- A regenerated phase-2 delta figure or paired-point figure for the same blocks
- A dedicated curriculum-ordering comparison figure (`Grouped->Random` vs `Random->Grouped`)
- A dedicated length-curriculum figure (`Easy->Hard` vs `Hard->Easy`, plus `exp_032` as an extended comparison)
- A generated generation-metrics comparison figure

## Appendix should include:

- Tables:
- Full per-seed optimization results
- Full per-seed generation metrics
- Archived block-wise summaries
- Files:
- Run manifests
- Run summaries and phase summaries
- Prediction CSVs
- `reports.zip`
- Extra plots:
- Random-baseline cross-block diagnostic plot
- Stale/duplicate plot snapshots from `mnt` and `exports`

## Practical thesis-load guidance

- Do not use the current `plot_5k_full_curriculum_comparison_eval_loss.png` as evidence; regenerate it from the current master summary.
- Do not use `plot_cross_block_eval_trends.png`; it is stale and incomplete.
- Use the current `reports\master\master_summary_table.csv` as the canonical optimization-results source.
- For generation metrics, cite the exact source file used because aggregated CSVs and per-seed JSONs conflict.
- Mixed batching should not be presented as an evaluated production strategy in the main thesis; it appears only in exploratory notebook material.
