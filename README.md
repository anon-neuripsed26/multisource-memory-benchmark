# survey2agent — diagnostic testbed for selective QA over multi-source personal memory

![code license](https://img.shields.io/badge/code-Apache--2.0-blue.svg)
![data license](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)
![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![status](https://img.shields.io/badge/status-double--blind%20review-yellow.svg)

> NeurIPS 2026 Datasets &amp; Benchmarks (Evaluations &amp; Datasets) Track submission.
> This repository ships the dataset, code, and reproducibility artifacts for the accompanying paper.

## Review artifacts

- **Code repository:** <https://github.com/anon-neuripsed26/multisource-memory-benchmark>
- **Hugging Face dataset and frozen outputs:** <https://huggingface.co/datasets/anon-neuripsed26/multisource-memory-benchmark>

---

## One-command reproduction

After cloning, every paper number can be reproduced from frozen LLM
outputs that are downloaded once from Hugging Face — **zero API
calls**, **zero spend** during reproduction.

```bash
make all          # install → fetch dataset → smoke test → reproduce 23 tables
```

`make all` runs in roughly 1–2 hours on an Apple M1 Pro laptop with
16 GB RAM (CPU-only), including install and network-dependent fetch time.
Each step can also be invoked individually:

```bash
make install      # pip install with the api/dev/hf extras  (~30 s)
make fetch        # download release ZIP from Hugging Face  (~1-3 min; ~36 MB, expands to ~410 MB)
make smoke        # end-to-end smoke test                   (~30 s)
make reproduce    # reproduce all 23 paper tables           (~40-60 min)
```

If you prefer not to use `make`, the equivalent shell incantations are:

```bash
python3 -m pip install --upgrade pip setuptools wheel
pip install -e '.[api,dev,hf]'
python data/fetch_benchmark.py
PYTHONPATH=src python -m pytest tests/test_smoke_end_to_end.py -v
PYTHONPATH=src python -m paper_artifacts.reproduce_paper
```

The shell-quoted form `'.[api,dev,hf]'` is required on zsh (default macOS
shell). On bash either form works. The pip upgrade is only needed if
your interpreter ships with `pip < 22` (which lacks PEP 621 editable
install support).

---

## What this repository contains

| Path | Responsibility |
|------|----------------|
| `src/survey2agent/` | Python package (data generation, extraction, a deployable method suite (T0 trivial through T3 LLM) + Source Reachability reference, selective layer, evaluation, CLI) |
| `configs/` | Single source of truth for model IDs (`models.yaml`), question spec (`questions.yaml`), seeds and splits, few-shot exemplars |
| `data/` | Hugging Face fetch script (`fetch_benchmark.py`) and a tiny offline sample (`data/sample/`); the full 4-seed × 480-persona benchmark plus frozen LLM outputs (`extracted_atoms/`, `method_outputs/`) are downloaded by `make fetch` from a checksum-verified release ZIP |
| `paper_artifacts/` | One reproduction script per paper table or figure; orchestrated by `reproduce_paper.py` |
| `results/` | Created by `make fetch` / `make reproduce`: fitted method parameters, final tables, SHA256 cache root for live API runs |
| `tests/` | Pytest suite (including end-to-end smoke tests); full pytest run is ~35-45 min, smoke alone ~30 s |
| `scripts/` | Cross-platform Python entry points for re-running individual stages |

For a one-page architecture overview see [ARCHITECTURE.md](ARCHITECTURE.md).
For the cache key format and reproducibility guarantees see [CACHE_POLICY.md](CACHE_POLICY.md).
For the full ML reproducibility checklist see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
For dataset provenance, intended use, and limitations see [DATASHEET.md](DATASHEET.md).

---

## Persona-ID information boundary

Persona IDs are bookkeeping keys, not model inputs. They are used for file
joins, cache keys, output filenames, and bootstrap grouping, but the
reference predictors and prompt builders do not pass the target persona's
ID, seed path, split path, or difficulty-prefix strings (for example
`bench_stable`, `bench_shift`, `bench_stated`) as prediction features or
LLM prompt text. DSNBF uses difficulty labels only where the paper says it
does: the official runner passes train/calibration difficulty metadata from
the persona spec for fitting per-difficulty matrices, and reporting uses
difficulty strata only after prediction.

If you add a method, do not parse `persona_id`, filenames, seed names, or
directory paths to infer difficulty. The method-facing payload is the atom
table, rendered memory, or structured atom grid only. Regression tests under
`tests/methods/test_prompt_persona_id_boundary.py` guard the LLM prompt side
of this contract. External evaluators should strip or ignore identifiers before
invoking prediction code; using difficulty prefixes as features is leakage, not
a valid benchmark submission.

---

## Extending the benchmark

This is a **living testbed**: adding your own method, question,
evidence stream, or LLM backend is first-class, not an afterthought.
All 13 shipped methods iterate `QUESTIONS.keys()`, so a new question
added via [example 02](examples/02_custom_question/) is automatically
evaluated by every method with zero changes outside
[`configs/questions.yaml`](configs/questions.yaml) and the
ground-truth rule file. Methods never hard-code question ids.

| Extension point | Worked example | Full reference | Effort |
|---|---|---|---|
| Add a new method | [examples/01_minimal_method/](examples/01_minimal_method/) (runnable) | [EXTENDING.md §1](EXTENDING.md#1-add-a-new-method) | ~30 min |
| **Add your own question** | [examples/02_custom_question/](examples/02_custom_question/) (runnable) | [EXTENDING.md §2](EXTENDING.md#2-add-a-new-question-type) | ~1 hour |
| Drive evaluation programmatically | [examples/03_programmatic_api/](examples/03_programmatic_api/) (runnable) | — | ~5 min |
| Plug a new LLM backend | — | [EXTENDING.md §5](EXTENDING.md#5-plug-in-a-new-llm-backend) | ~1 hour |
| Add a new evidence stream | [examples/05_custom_stream/](examples/05_custom_stream/) (runnable sandbox; production integration checklist in EXTENDING §4) | [EXTENDING.md §4](EXTENDING.md#4-add-a-new-evidence-stream) | ~1 day |
| Reproduce a paper-locked table | [paper_artifacts/appendix/source_ceiling_complement_table.py](paper_artifacts/appendix/source_ceiling_complement_table.py) | [EXTENDING.md §1.7](EXTENDING.md#17-reproduce-a-paper-locked-appendix-table) | ~1 hour |

The runnable examples are guarded by `make smoke` — CI fails if any
extension-point contract breaks.

---

## Paper claim → script map

`make reproduce` runs all 23 paper scripts below in sequence. Each script
writes a `<table>.csv` and `<table>.md` pair under
`paper_artifacts/output/{main,appendix}/`, with a per-cell `paper_match`
column comparing computed values against paper-locked constants at
±0.005 absolute tolerance (≈ ±0.5 pp). A non-zero exit code from any
script signals tolerance drift.

Most scripts recompute paper cells directly from benchmark records, direct
readout atoms, and frozen LLM outputs. The Appendix-F robustness scripts read
aggregate JSONs under `data/benchmark/results/`; those aggregate JSONs are
also reproducible from the released data with:

```bash
make regenerate-appendix-f-results   # ~45-60 min, no API calls
# Equivalent alias retained for older review notes:
make regenerate-appendix-c-results
```

This upstream target rebuilds the training-size, extraction-noise, DGP
perturbation, cross-seed/cross-extractor, and per-question extraction-accuracy
JSONs used by the Appendix-F paper-lock scripts.
In a recent fresh-clone audit, regenerating these aggregates followed by
`make reproduce-appendix` produced 0 failed paper-lock cells.

Artifacts are identified by **PDF section name + caption descriptor**, not by
table number or LaTeX label letter — both can shift between recompiles, and
the mapping below is stable as long as the section titles and table captions
are stable.

### Main paper artifacts (4 scripts)

| PDF section · Artifact | Script | Wall clock |
|------------------------|--------|------------|
| Experiments · Macro accuracy under answer-only mode (4-seed pooled, 95% bootstrap CI) | [`main.forced_accuracy_main`](paper_artifacts/main/forced_accuracy_main.py) | ~10–15 min |
| Experiments · 2×2 factorial decomposition of the Fusion-vs-LLM gap (this is a *figure* in the PDF; the script also writes the underlying table CSV) | [`main.factorial_decomposition`](paper_artifacts/main/factorial_decomposition.py) | ~3–5 min |
| Experiments · Accuracy by reasoning type with paired Δ (DSNBF − GPT-μ*) | [`main.per_type_accuracy`](paper_artifacts/main/per_type_accuracy.py) | ~12–18 min |
| Selective QA Details and Few-Shot Supplementary · Full selective QA table (per-seed-then-mean, no bootstrap) | [`main.selective_qa_full`](paper_artifacts/main/selective_qa_full.py) | ~1–2 min |

### Appendix artifacts (19 scripts)

| PDF section · Artifact | Script |
|------------------------|--------|
| Benchmark Details and Ground-Truth Verification · Atom Extraction Faithfulness Audit (audits inline prose claims in this subsection — no standalone table number in the PDF) | [`appendix.atom_extraction_faithfulness`](paper_artifacts/appendix/atom_extraction_faithfulness.py) |
| Benchmark Details and Ground-Truth Verification · Source-Reachability Complement (answer-only accuracy on Full / GT-present / GT-absent slices, 19 methods × 3 metrics = 57 paper-lock cells) | [`appendix.source_ceiling_complement_table`](paper_artifacts/appendix/source_ceiling_complement_table.py) |
| Method Details · Compute Footprint — API call inventory and cache cross-check | [`appendix.api_cache_inventory`](paper_artifacts/appendix/api_cache_inventory.py) |
| Extended Diagnostic Tables · Per-Type Accuracy — Full per-type macro accuracy | [`appendix.per_type_macro_accuracy_full`](paper_artifacts/appendix/per_type_macro_accuracy_full.py) |
| Extended Diagnostic Tables · Difficulty-Class Breakdown — T2 fusion by reasoning type × difficulty (forced + selective) | [`appendix.t2_fusion_per_type_per_difficulty`](paper_artifacts/appendix/t2_fusion_per_type_per_difficulty.py), [`appendix.t2_fusion_selective_per_type_per_difficulty`](paper_artifacts/appendix/t2_fusion_selective_per_type_per_difficulty.py) |
| Extended Diagnostic Tables · Difficulty-Class Breakdown — T3 LLM by reasoning type × difficulty (forced + selective) | [`appendix.t3_llm_per_type_per_difficulty`](paper_artifacts/appendix/t3_llm_per_type_per_difficulty.py), [`appendix.t3_llm_selective_per_type_per_difficulty`](paper_artifacts/appendix/t3_llm_selective_per_type_per_difficulty.py) |
| Extended Diagnostic Tables · Prediction Distributions on Failure Questions — E-factor (E1 / E2) histograms | [`appendix.prediction_distributions_e_causal`](paper_artifacts/appendix/prediction_distributions_e_causal.py) |
| Extended Diagnostic Tables · Prediction Distributions on Failure Questions — C-plan-reality / F-missing-data (C2 / F3) histograms | [`appendix.prediction_distributions_c_pr_f_miss`](paper_artifacts/appendix/prediction_distributions_c_pr_f_miss.py) |
| Selective QA Details and Few-Shot Supplementary · Few-Shot Supplementary Check (single seed) | [`appendix.few_shot_supplementary`](paper_artifacts/appendix/few_shot_supplementary.py) |
| Additional Robustness Analyses · Cross-Seed Stability | [`appendix.cross_seed_stability`](paper_artifacts/appendix/cross_seed_stability.py) |
| Additional Robustness Analyses · Training Size Sensitivity | [`appendix.train_size_ablation`](paper_artifacts/appendix/train_size_ablation.py) |
| Additional Robustness Analyses · GPT-5.4 vs. Gemini Cross-Condition Comparison | [`appendix.cross_condition_gpt_vs_gemini`](paper_artifacts/appendix/cross_condition_gpt_vs_gemini.py) |
| Additional Robustness Analyses · DGP Perturbation (9×7 grid) | [`appendix.dgp_perturbation`](paper_artifacts/appendix/dgp_perturbation.py) |
| Additional Robustness Analyses · Extraction Noise Tolerance (full analysis) | [`appendix.noise_perturbation`](paper_artifacts/appendix/noise_perturbation.py) |
| Additional Robustness Analyses · Cross-Extractor Robustness | [`appendix.cross_extractor_robustness`](paper_artifacts/appendix/cross_extractor_robustness.py) |
| Additional Robustness Analyses · Cross-Parameter Transfer Without Refit (DSNBF) | [`appendix.cross_bias_transfer`](paper_artifacts/appendix/cross_bias_transfer.py) |
| Additional Robustness Analyses · Cross-Extractor Robustness — Per-question × per-difficulty extraction accuracy | [`appendix.per_question_extraction_accuracy`](paper_artifacts/appendix/per_question_extraction_accuracy.py) |

To run a subset:

```bash
PYTHONPATH=src python -m paper_artifacts.reproduce_paper --tier main
PYTHONPATH=src python -m paper_artifacts.reproduce_paper --names few_shot_supplementary
```

Per-script details (caption, paper source, reduction, methods, tolerance)
are in [`paper_artifacts/MANIFEST.md`](paper_artifacts/MANIFEST.md).

---

## Re-running the LLM pipeline (optional, costs API spend)

The frozen artifacts under `data/method_outputs/` reproduce every paper
number with no API access. If you want to regenerate them — for a new
model, prompt variant, or seed — five producer subcommands are exposed
through the `survey2agent` CLI:

```bash
PYTHONPATH=src python -m survey2agent run-extraction --help
PYTHONPATH=src python -m survey2agent run-llm-direct --help
PYTHONPATH=src python -m survey2agent run-schema-aware --help
PYTHONPATH=src python -m survey2agent run-struct-llm --help
PYTHONPATH=src python -m survey2agent run-few-shot --help
```

Each subcommand requires `--provider`, `--model` (must be a key in
[`configs/models.yaml`](configs/models.yaml)), `--seed`, `--personas`
(file or `all`), and `--output-dir`. By default the runner is
**cache-only**: requests that miss the SHA256 cache at
`results/released/cached_api_outputs/` raise rather than touch the
network. Pass `--allow-api-call` and set the relevant API key
(`OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`) to make live
calls. See `--help` on each subcommand for the full argument list.
For batch providers (OpenAI and Gemini), cache-only mode is a no-network
guardrail rather than a frozen-output replay interface: without
`--allow-api-call`, a cache miss exits cleanly instead of submitting a batch.
Paper reproduction uses the frozen artifacts through `paper_artifacts/`, not
through these live producer commands.

For few-shot regeneration, pass the shipped prompt bundle explicitly if
running outside the repository root:

```bash
PYTHONPATH=src python -m survey2agent run-few-shot \
  --provider openai --model gpt-5.4 \
  --seed data/benchmark/seeds/s20260321 \
  --personas /tmp/s2a_personas.txt \
  --questions all \
  --configs-root configs/few_shot \
  --output-dir /tmp/s2a_few_shot
```

---

## Licenses

- Code: [Apache License 2.0](LICENSE)
- Data: [Creative Commons Attribution 4.0 International](DATA_LICENSE)

## Citation

See [`CITATION.cff`](CITATION.cff). The canonical citation will be
updated at camera-ready.

## Anonymization

This repository is anonymized for double-blind review. Author identities,
lab affiliations, and any internal codenames have been removed. The
anonymous review mirror is hosted at
<https://github.com/anon-neuripsed26/multisource-memory-benchmark>
and the dataset/frozen-output release is hosted at
<https://huggingface.co/datasets/anon-neuripsed26/multisource-memory-benchmark>
(both are also linked from the OpenReview submission).
