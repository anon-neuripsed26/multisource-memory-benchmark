# paper_artifacts/

User-facing reproducibility entry points for the Survey2Agent paper
(NeurIPS 2026 E&D submission). Scripts read frozen LLM outputs from
`data/method_outputs/` and oracle atoms from
`data/benchmark/seeds/<seed>/`, then write a CSV + Markdown
pair under `output/<tier>/` reproducing one paper table to within
±0.005 absolute (±0.5 pp on the percentage scale).

**No LLM API calls** — everything runs from the frozen artifacts
shipped with this repository.

## Layout

```
paper_artifacts/
  ├─ _common.py               shared helpers (loaders, runners, output writers)
  ├─ reproduce_paper.py       CLI orchestrator (--tier {main,appendix,all})
  ├─ MANIFEST.md              per-script paper mapping (caption + reduction + tolerance)
  ├─ main/                    scripts that reproduce the four main paper tables
  │   ├─ forced_accuracy_main.py     →  paper Forced-Accuracy Main Table
  │   ├─ factorial_decomposition.py  →  paper 2x2 Factorial Decomposition
  │   ├─ per_type_accuracy.py        →  paper Per-Type Diagnostic Analysis
  │   └─ selective_qa_full.py        →  paper Full Selective QA Table (appendix table; lives
  │                                       in main/ for runtime convenience)
  ├─ appendix/                per-type / per-difficulty breakdowns, robustness, prediction distributions, supplementary checks (18 scripts — see the table below)
  └─ output/
      ├─ main/                CSV + MD outputs from main scripts
      └─ appendix/            CSV + MD outputs from appendix scripts
```

## Scripts

Main paper tables (4):

| Script                             | Reproduces      | Reduction                              | Wall clock |
|:-----------------------------------|:----------------|:---------------------------------------|:-----------|
| `main.forced_accuracy_main`        | Forced-Accuracy Main Table           | pool-then-bootstrap (B=2000, persona)  | ~10–15 min |
| `main.factorial_decomposition`     | 2x2 Factorial Decomposition           | pool-then-bootstrap (B=2000, persona)  | ~3–5 min   |
| `main.per_type_accuracy`           | Per-Type Diagnostic Analysis           | per-type pool-then-bootstrap + paired Δ| ~12–18 min |
| `main.selective_qa_full`           | Full Selective QA Table         | per-seed-then-mean (no bootstrap)      | ~1–2 min   |

Appendix tables (19):

| Script                             | Reproduces      | Reduction                              | Wall clock |
|:-----------------------------------|:----------------|:---------------------------------------|:-----------|
| `appendix.atom_extraction_faithfulness`             | Atom Extraction Faithfulness Audit| per-seed-then-mean over extraction audit | ~1 min   |
| `appendix.source_ceiling_complement_table`          | Source-Reachability Complement      | full / GT-present / GT-absent slice acc  | ~2–3 min |
| `appendix.cross_seed_stability`                     | Cross-Seed Stability       | per-seed point + cross-seed std         | ~1–2 min |
| `appendix.train_size_ablation`                      | Training Size Sensitivity       | per-train-size sweep                    | ~3–4 min |
| `appendix.cross_condition_gpt_vs_gemini`            | GPT-5.4 vs Gemini Cross-Condition       | seed-1 paired GPT vs Gemini             | ~1 min   |
| `appendix.dgp_perturbation`                         | DGP Perturbation       | 9×7 perturbation grid                  | <30 s    |
| `appendix.noise_perturbation`                       | Extraction Noise Tolerance       | noise grid                              | <30 s    |
| `appendix.cross_extractor_robustness`               | Cross-Extractor Robustness       | extractor swap (GPT-Gem)                | <30 s    |
| `appendix.cross_bias_transfer`                      | —            | DSNBF cross-parameter transfer (no refit) vs target-retrained, 9 b×d variants | ~9 min   |
| `appendix.per_question_extraction_accuracy`         | Per-Question Extraction Accuracy      | per-Q × per-diff extraction accuracy    | <30 s    |
| `appendix.per_type_macro_accuracy_full`             | Per-Type Accuracy       | per-seed-then-mean per type             | ~2–3 min |
| `appendix.t2_fusion_per_type_per_difficulty`        | Difficulty-Class Breakdown (T2 fusion)      | per-seed-then-mean per (type, diff)     | ~1–2 min |
| `appendix.t3_llm_per_type_per_difficulty`           | Difficulty-Class Breakdown (T3 LLM)      | per-seed-then-mean per (type, diff)     | <30 s    |
| `appendix.t2_fusion_selective_per_type_per_difficulty` | Difficulty-Class Breakdown (T2 +SKIP) | pool-then-micro per (type, diff)     | ~1–2 min |
| `appendix.t3_llm_selective_per_type_per_difficulty`    | Difficulty-Class Breakdown (T3 +SKIP) | pool-then-micro per (type, diff)     | <30 s    |
| `appendix.prediction_distributions_e_causal`        | Prediction Distributions (E-type)       | E1/E2 prediction histograms             | <30 s    |
| `appendix.prediction_distributions_c_pr_f_miss`     | Prediction Distributions (C/F-type)      | C2/F3 prediction histograms             | <30 s    |
| `appendix.few_shot_supplementary`                   | App F.3      | seed-1 few-shot vs Direct vs DSNBF      | <30 s    |
| `appendix.api_cache_inventory`                      | API Call Inventory       | per-stage × vendor call counts          | <30 s    |

Orchestrator: `reproduce_paper` runs all 23 in sequence (~40–60 min on an
Apple M1 Pro laptop with 16 GB RAM, CPU-only).

**Reduction note**: answer-only appendix tables (D1, E2a,
E2b) use **per-seed-then-mean** per cell — in answer-only mode, all
per-seed counts are equal (coverage = 100%), so per-seed mean and
pooled micro coincide. Selective tables (E2a-skip, E2b-skip) use
**pool-then-micro**: pool results across the 4 seeds, group by
`(reasoning_type, difficulty_class)`, then compute micro selective
accuracy. Coverage varies across seeds for selective methods, so the
two reductions diverge and the paper's published numbers match the
pooled-micro form (verified against `data/benchmark/results/d2_skip_perq_perdiff_4seed.json`).

## Quick start

```bash
# from repository root

# all 23 tables in one command (~40-60 min)
make reproduce

# 4 main tables only
PYTHONPATH=src python -m paper_artifacts.reproduce_paper --tier main

# 19 appendix tables only
PYTHONPATH=src python -m paper_artifacts.reproduce_paper --tier appendix

# subset by name (comma-separated)
PYTHONPATH=src python -m paper_artifacts.reproduce_paper \
    --names forced_accuracy_main,factorial_decomposition

# fastest single table (no bootstrap, no per-type)
PYTHONPATH=src python -m paper_artifacts.main.selective_qa_full

# subset of rows within a single script
PYTHONPATH=src python -m paper_artifacts.main.selective_qa_full \
    --rows ssb_ext,dsnbf_oracle
```

## Output

Each script writes two files into `paper_artifacts/output/<tier>/`:

* `<table>.csv` — 13-column machine-readable record:
  `row_id, method_label, mode, metric, point, ci_low, ci_high, n_seeds,
  n_personas, paper_value_low, paper_value_point, paper_value_high,
  paper_match`.
  `paper_match` is `OK` / `FAIL d=X.XXXX` / `NA` (when the paper does
  not report a comparable cell, e.g. LLM rows in Forced-Accuracy Main Table quoted as point only).
* `<table>.md` — human-readable Markdown table mirroring the paper
  layout, with a header comment containing timestamp, tolerance, and
  paper source path.

## Conventions

* **Reductions**:
  * Forced-Accuracy Main Table / 2x2 Factorial Decomposition / Per-Type Diagnostic Analysis → `aggregate_pooled_with_ci(metric="forced_accuracy",
    n_bootstrap=2000, seed=42, cluster_by="persona")`. 95% percentile CI.
  * Full Selective QA Table → `aggregate_per_seed_then_average(metric=...)` (per-seed
    macro, then arithmetic mean across the four canonical seeds).
* **Tolerance**: ±0.005 absolute (±0.5 pp). Cells outside the band
  surface as `FAIL d=...` in the CSV and increment the script's exit code.
* **Random row determinism**: Forced-Accuracy Main Table `Random` is constructed with
  `seed=42` so the point estimate is reproducible across runs (paper
  was also seeded; the ±0.005 tolerance accommodates a different seed).

## ArgRAG-Style Adaptation

The paper's equal-weight closed-class ArgRAG adaptation and Majority Vote
are provably identical on the 5-source atom input studied here (paper §3).
Forced-Accuracy Main Table lists them on
a single shared row labelled `Majority Vote / ArgRAG-style` and reuses the
`MajorityVote` computation. No full document-level `ArgRAG` class is wired.

## Struct-LLM

2x2 Factorial Decomposition and Per-Type Diagnostic Analysis require GPT-5.4 reading structured atoms (oracle and
extracted) as prompt. The 960 frozen artifacts (4 seeds × 2 modes ×
120 personas) live at
`data/method_outputs/struct_llm/{mode}/{seed}/<persona>.json`
and are exposed via the new `StructLLMSource` (`survey2agent.methods`).
The artifact schema is byte-identical to `FrozenBulkJSONSource`.
