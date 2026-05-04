# MANIFEST — paper table reproduction scripts

This manifest maps each released reproduction script to the paper result it
checks. It is intentionally self-contained for the anonymous code artifact:
all script paths below exist in this repository, and paper locations are given
by section/table names rather than local author-side LaTeX paths.

All scripts read the fetched data bundle under `data/` (or `S2A_DATA_ROOT`) and
write CSV/Markdown outputs to `paper_artifacts/output/`. They do not make live
LLM API calls. A cell is marked `OK` when the regenerated value is within the
script's stated tolerance of the paper-locked value.

## Main Tier

### `paper_artifacts/main/forced_accuracy_main.py`

- Paper result: main answer-only accuracy table.
- Paper location: Experiments, "Main Results and Selective QA".
- Reduction: 4-seed pooled macro accuracy with persona-cluster bootstrap CIs.
- Inputs: released benchmark labels, canonical GPT-extracted atoms, direct-readout atoms, and frozen LLM outputs.
- Key rows: Random, Majority Class, SSB, Majority Vote / ArgRAG-style adaptation, BCF, NBF, DSNBF, ABF, four LLM families, and Source Reachability reference.
- Tolerance: +/-0.005 on point estimates and CI endpoints.

### `paper_artifacts/main/factorial_decomposition.py`

- Paper result: 2x2 resolver-by-input decomposition figure.
- Paper location: Experiments, "Method-vs-input decomposition".
- Reduction: matched-input macro accuracy with persona-cluster bootstrap CIs.
- Inputs: DSNBF and GPT-5.4 structured-input diagnostic under direct-readout and extracted-atom conditions.
- Tolerance: +/-0.005 on accuracy cells and +/-0.5 pp on decomposition percentages.

### `paper_artifacts/main/per_type_accuracy.py`

- Paper result: reasoning-type diagnostic accuracy table.
- Paper location: Experiments, "Diagnostic Analysis by Reasoning Type".
- Reduction: 4-seed pooled macro accuracy by reasoning type plus paired bootstrap deltas.
- Inputs: DSNBF on direct-readout atoms, GPT-5.4 on matched structured atoms, prompt-only LLM families, and Source Reachability reference.
- Tolerance: +/-0.005 on per-type points and deltas.

### `paper_artifacts/main/selective_qa_full.py`

- Paper result: full selective QA table.
- Paper location: Appendix, "Full Selective QA Table".
- Reduction: mean of per-seed selective accuracy and coverage.
- Inputs: selective fusion methods on extracted/direct-readout atoms and LLM self-abstention outputs.
- Tolerance: +/-0.005 on selective accuracy and coverage cells.

## Appendix Tier

### `paper_artifacts/appendix/per_type_macro_accuracy_full.py`

- Paper result: full per-type macro accuracy grid.
- Paper location: Appendix, "Per-Type Accuracy".
- Reduction: per-seed then mean by reasoning type.
- Inputs: fusion/single-source methods and prompt-only LLM methods.
- Tolerance: +/-0.005 on every cell.

### `paper_artifacts/appendix/t2_fusion_per_type_per_difficulty.py`

- Paper result: T2 fusion per-type by difficulty grid, answer-only mode.
- Paper location: Appendix, "Difficulty-Class Breakdown, T2 Fusion".
- Reduction: per-seed then mean by reasoning type and difficulty class.
- Inputs: DSNBF, NBF, ABF, SSB, BCF, and Majority Vote.
- Tolerance: +/-0.005 on cells and footer drops.

### `paper_artifacts/appendix/t3_llm_per_type_per_difficulty.py`

- Paper result: T3 LLM per-type by difficulty grid, answer-only mode.
- Paper location: Appendix, "Difficulty-Class Breakdown, T3 LLM".
- Reduction: per-seed then mean by reasoning type and difficulty class.
- Inputs: Direct and Schema-Aware outputs for GPT-5.4, Gemini, DeepSeek, and Qwen3.
- Tolerance: +/-0.005 on cells and footer drops.

### `paper_artifacts/appendix/t2_fusion_selective_per_type_per_difficulty.py`

- Paper result: T2 fusion selective-accuracy/coverage grid.
- Paper location: Appendix, "Difficulty-Class Breakdown, T2 +SKIP".
- Reduction: 4-seed pooled micro selective accuracy and coverage by reasoning type and difficulty class.
- Inputs: DSNBF+SKIP, NBF+SKIP, ABF+SKIP, SSB+SKIP, BCF, and Majority Vote.
- Tolerance: +/-0.005 on cells and footer drops.

### `paper_artifacts/appendix/t3_llm_selective_per_type_per_difficulty.py`

- Paper result: T3 LLM selective-accuracy/coverage grid.
- Paper location: Appendix, "Difficulty-Class Breakdown, T3 +SKIP".
- Reduction: 4-seed pooled micro selective accuracy and coverage by reasoning type and difficulty class.
- Inputs: Direct and Schema-Aware self-abstention outputs for GPT-5.4, Gemini, DeepSeek, and Qwen3.
- Tolerance: +/-0.005 on cells and footer drops.

### `paper_artifacts/appendix/prediction_distributions_e_causal.py`

- Paper result: E1/E2 prediction distributions by difficulty class.
- Paper location: Appendix, "Prediction Distributions on Failure Questions".
- Reduction: answer histograms and accuracy rows for selected questions/difficulties.
- Inputs: GT, GPT-5.4 Schema-Aware, DeepSeek Schema-Aware, and Qwen3 Schema-Aware outputs.
- Tolerance: +/-0.005 on non-empty cells; paper dash cells are skipped.

### `paper_artifacts/appendix/prediction_distributions_c_pr_f_miss.py`

- Paper result: C2/F3 prediction distributions by difficulty class.
- Paper location: Appendix, "Prediction Distributions on Failure Questions".
- Reduction: answer histograms and accuracy rows for selected questions/difficulties.
- Inputs: GT, GPT-5.4 Schema-Aware, DeepSeek Schema-Aware, and Qwen3 Schema-Aware outputs.
- Tolerance: +/-0.005 on non-empty cells; paper dash cells are skipped.

### `paper_artifacts/appendix/cross_seed_stability.py`

- Paper result: cross-seed stability table.
- Paper location: Appendix, "Cross-Seed Stability".
- Reduction: per-seed scores and 4-seed mean/std summaries from released aggregate JSONs.
- Inputs: `data/benchmark/results/` aggregate files.
- Tolerance: +/-0.005 on paper-locked cells.

### `paper_artifacts/appendix/train_size_ablation.py`

- Paper result: training-size sensitivity table.
- Paper location: Appendix, "Training Size Sensitivity".
- Reduction: 4-seed mean/std over deterministic training-size sweeps.
- Inputs: `data/benchmark/results/` aggregate files regenerated by `make regenerate-appendix-f-results`.
- Tolerance: +/-0.005 on paper-locked cells.

### `paper_artifacts/appendix/cross_condition_gpt_vs_gemini.py`

- Paper result: GPT-5.4 vs Gemini cross-condition comparison.
- Paper location: Appendix, "GPT-5.4 vs Gemini Cross-Condition Comparison".
- Reduction: seed-1 condition grid from frozen GPT/Gemini extraction and method-output artifacts.
- Inputs: `data/extracted_atoms/` and `data/method_outputs/gemini_p2/`.
- Tolerance: +/-0.005 on paper-locked cells.

### `paper_artifacts/appendix/dgp_perturbation.py`

- Paper result: DGP perturbation grid over bias and dropout scales.
- Paper location: Appendix, "DGP Perturbation".
- Reduction: 4-seed mean/std from released aggregate JSONs.
- Inputs: `data/benchmark/results/` aggregate files regenerated by `make regenerate-appendix-f-results`.
- Tolerance: +/-0.005 on paper-locked cells.

### `paper_artifacts/appendix/noise_perturbation.py`

- Paper result: extraction noise tolerance table.
- Paper location: Appendix, "Extraction Noise Tolerance".
- Reduction: 4-seed mean/std from released aggregate JSONs.
- Inputs: `data/benchmark/results/` aggregate files regenerated by `make regenerate-appendix-f-results`.
- Tolerance: +/-0.005 on paper-locked cells.

### `paper_artifacts/appendix/cross_extractor_robustness.py`

- Paper result: GPT vs Gemini extractor robustness table.
- Paper location: Appendix, "Cross-Extractor Robustness".
- Reduction: 4-seed mean/std from released GPT/Gemini atom artifacts and aggregate JSONs.
- Inputs: canonical GPT atoms in `data/extracted_atoms/` and Gemini atoms in `data/method_outputs/gemini_p2/`.
- Tolerance: +/-0.005 on paper-locked cells.

### `paper_artifacts/appendix/cross_bias_transfer.py`

- Paper result: cross-parameter transfer without refit.
- Paper location: Appendix, "Cross-Parameter Transfer Without Refit".
- Reduction: recomputes source projections and GT labels under bias/dropout scale variants, then evaluates DSNBF transfer vs target-retrained arms.
- Inputs: released benchmark seeds plus deterministic source-projection code; no LLM artifacts.
- Tolerance: +/-0.005 on accuracy means and +/-0.5 pp on gap columns.

### `paper_artifacts/appendix/per_question_extraction_accuracy.py`

- Paper result: per-question, per-difficulty extraction accuracy subtable.
- Paper location: Appendix, "Cross-Extractor Robustness".
- Reduction: compares extracted atoms against direct-readout atoms by question and difficulty.
- Inputs: canonical GPT atoms, Gemini atoms, and direct-readout atoms.
- Tolerance: +/-0.005 on paper-locked cells.

### `paper_artifacts/appendix/atom_extraction_faithfulness.py`

- Paper result: atom extraction faithfulness audit reported in appendix prose.
- Paper location: Appendix, "Atom Extraction Faithfulness Audit".
- Reduction: cell-level comparison between extracted atoms and direct-readout atoms, including null/null equality where stated.
- Inputs: canonical GPT atoms and direct-readout atoms.
- Tolerance: +/-0.005 on paper-locked prose values.

### `paper_artifacts/appendix/source_ceiling_complement_table.py`

- Paper result: Source Reachability complement diagnostic table.
- Paper location: Appendix, "GT Construction Notes".
- Reduction: partitions test instances by whether any direct-readout source atom exactly equals GT, then scores methods on full / GT-present / GT-absent slices.
- Inputs: direct-readout atoms, frozen method outputs, and GT labels.
- Tolerance: +/-0.005 on every cell.

### `paper_artifacts/appendix/api_cache_inventory.py`

- Paper result: compute footprint and released-cache inventory cross-check.
- Paper location: Appendix, "Compute Footprint".
- Reduction: validates released frozen-output counts against the paper's API-workload/cache-coverage description.
- Inputs: `data/method_outputs/`, `data/extracted_atoms/`, and released cache metadata.
- Tolerance: exact integer checks for inventory counts; paper-point checks where applicable.

### `paper_artifacts/appendix/few_shot_supplementary.py`

- Paper result: single-seed few-shot supplementary check.
- Paper location: Appendix, "Few-Shot Supplementary Check".
- Reduction: compares GPT-5.4 few-shot outputs against direct prompting on seed `s20260321`.
- Inputs: frozen few-shot and direct outputs under `data/method_outputs/gpt-5.4/s20260321/`.
- Tolerance: +/-0.005 on paper-locked cells.

## Regeneration Notes

- `make reproduce` runs all main and appendix scripts above.
- `make reproduce-main` runs the four main-tier scripts.
- `make reproduce-appendix` runs the nineteen appendix-tier scripts.
- `make regenerate-appendix-f-results` rebuilds the aggregate JSON files used by the robustness, perturbation, and ablation scripts from released data and frozen artifacts.
- The compatibility alias `make regenerate-appendix-c-results` intentionally points to `make regenerate-appendix-f-results` for reviewers following an earlier appendix-letter naming convention.
