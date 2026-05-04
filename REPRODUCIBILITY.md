# Reproducibility Checklist (NeurIPS 2026 E&D Track)

This document follows the [ML Reproducibility Checklist](https://arxiv.org/abs/2003.12206) and lists how this repository satisfies each item.

---

## For all models and algorithms

- [x] **A clear description of the mathematical setting, algorithm, and/or model.**
  See paper §2 (problem formulation), §3 (method comparison), Appendix C (per-method details).
- [x] **A clear explanation of any assumptions.**
  Spec-level distortion assumptions are in paper Appendix B (per-source bias profile, missing-data rates).
- [x] **An analysis of the complexity (time, space, sample size) of any algorithm.**
  Paper §Compute Footprint reports cumulative wall-clock for deterministic methods and total LLM API calls (Appendix Table 13).

## For any theoretical claim

- [x] **A clear statement of the claim.**
  Paper §3 and Appendix C.
- [x] **A complete proof of the claim.**
  N/A — the paper does not advance new theoretical claims; it presents an empirical diagnostic comparison.

## For all datasets used

- [x] **The relevant statistics, such as number of examples.**
  4 seeds × 480 personas × 18 questions = 34,560 (persona, question) instances. Splits: 216 train / 48 dev / 96 cal / 120 test (45 / 10 / 20 / 25 %). 3 difficulty classes × 160 personas each.
- [x] **The details of train / validation / test splits.**
  See `configs/splits.yaml` and Appendix B.
- [x] **An explanation of any data that were excluded, and all pre-processing steps.**
  See Appendix B and `src/survey2agent/data_generation/`.
- [x] **A link to a downloadable version of the dataset or simulation environment.**
  Hugging Face: (anonymized for review). Local sample: `data/sample/`.
- [x] **For new data collected, a complete description of the data collection process.**
  See [DATASHEET.md](DATASHEET.md) and Appendix B (synthetic generation pipeline).

### Method-facing information boundary

Persona identifiers remain in files so the released bundle can be joined,
cached, audited, and bootstrapped reproducibly. They are not prediction
features. The reference evaluation path passes methods only the source atom
table (T1/T2), rendered NL memory (T3), or structured atom grid for the
matched-input diagnostic. LLM prompts exclude the target persona's ID, seed
name, split path, and difficulty-prefix strings such as `bench_stable`,
`bench_shift`, and `bench_stated`.

Difficulty labels are used for generation, fitting/calibration labels where
specified, few-shot exemplar selection, and after-the-fact reporting strata.
For DSNBF, the official runner passes train/calibration difficulty metadata
from the persona spec; reference methods do not need to parse difficulty from
persona IDs. Difficulty labels are not supplied as target-persona inputs at
prediction time. External methods should follow the same rule: do not parse
`persona_id`, filenames, or directory paths to infer difficulty. The regression test
`tests/methods/test_prompt_persona_id_boundary.py` checks the prompt-facing
side of this contract. External evaluator wrappers should strip or ignore
identifiers before invoking prediction code; using difficulty prefixes as
features is leakage, not a valid benchmark submission.

## For all shared code related to this work

- [x] **Specification of dependencies.**
  See `requirements.txt` and `pyproject.toml`.
- [x] **Training code.**
  Fitting routines are in `src/survey2agent/methods/<method>.py` (e.g., MLE for ABF parameters).
- [x] **Evaluation code.**
  See `src/survey2agent/evaluation/`.
- [x] **(Pre-)trained model(s).**
  Frozen LLM outputs are released under `data/method_outputs/`; deterministic fitted parameters and calibrated thresholds are recomputed by `make reproduce` from the released train/cal splits.
- [x] **README file includes table of results accompanied by precise commands to run to produce those results.**
  See [README.md](README.md) "Paper claim → script map" and `make reproduce` for the full pipeline.

## For all reported experimental results

- [x] **The range of hyper-parameters considered, the method to select the best hyper-parameter configuration, and the specification of all hyper-parameters used to generate results.**
  See `configs/` and Appendix C. Calibration was on the cal split (96 personas) using F0.5.
- [x] **The exact number of training and evaluation runs.**
  4 seeds × 32 method-modes × 18 questions × 480 personas. Bootstrap CI uses 2,000 resamples per method-mode. Few-shot is single-seed only (paper Appendix D).
- [x] **A clear definition of the specific measure or statistics used to report results.**
  Accuracy (top-1 against deterministic ground-truth answer), selective accuracy (accuracy among answered instances), coverage (fraction answered), F0.5. Bootstrap CIs at 95% level.
- [x] **A description of results with central tendency and variation.**
  See paper Tables 4-6 and Appendix Tables E2a-skip / E2b-skip with point estimates and CIs.
- [x] **The average runtime for each result, or estimated energy cost.**
  Paper §Compute Footprint reports total wall-clock and LLM API call counts. Per-stage API call counts are in Appendix Table 13. Cost was not tracked at runtime and is not reported.
- [x] **A description of the computing infrastructure used.**
  Single workstation (32 GB RAM, no GPU). Vendor LLM APIs for all extraction and LLM baselines.

---

## How to reproduce the paper end-to-end

### Layer A — Artifact reproducibility (no API spend, mandatory)

```bash
pip install -e '.[hf]'
python data/fetch_benchmark.py                # ≈36 MB ZIP from HF; expands to ≈410 MB
python -m paper_artifacts.reproduce_paper     # rebuilds all tables / figures from cached outputs
```

This regenerates every paper number from the frozen files under `data/extracted_atoms/`, `data/method_outputs/`, and `data/benchmark/results/`. **Zero API calls** are made. The output files under `paper_artifacts/output/` list every paper-reported value alongside the reproduced value for direct comparison.

By default, `fetch_benchmark.py` downloads the checksum-verified archive
`archives/multisource-memory-benchmark-data-v0.1.0.zip` and extracts it
locally. This is the reviewer-friendly path because the expanded HF tree has
about 29k small JSON files and can be rate-limited in unauthenticated
environments. To inspect or debug the expanded tree directly, use
`S2A_FETCH_MODE=snapshot python data/fetch_benchmark.py`.

The Appendix-F robustness tables use aggregate result JSONs in `data/benchmark/results/`. To rebuild those aggregates from the released benchmark, direct-readout atoms, GPT extracted atoms, and Gemini extracted atoms before running the paper-lock checks:

```bash
make regenerate-appendix-f-results                # ≈45-60 min, zero API calls
# Equivalent alias retained for older review notes:
make regenerate-appendix-c-results
python -m paper_artifacts.reproduce_paper --tier appendix
```

Recent fresh-clone audit results: `make fetch`, `make smoke`,
`make reproduce-main`, `make reproduce-appendix`, full `pytest tests/ -q`,
and `make regenerate-appendix-f-results` all completed without paper-lock
failures or test failures. The aggregate-regeneration target was followed by
`make reproduce-appendix`, which again reported 0 failed appendix cells.
Wall-clock varies by CPU and network, but the intended reviewer budget is
roughly 40-60 minutes for `make reproduce`, 45-60 minutes for Appendix-F
aggregate regeneration, and 35-45 minutes for the full pytest suite.

### Layer B — Pipeline reproducibility (small API spend, optional)

```bash
# Re-generate dataset for one seed (deterministic, ~5-10 min)
make seed-s20260321

# Optional live-call smoke on the shipped sample (smallest synchronous path:
# one persona × one question; costs API spend)
export OPENROUTER_API_KEY=...
PYTHONPATH=src python -m survey2agent run-few-shot \
  --provider openrouter --model deepseek-v3.2 \
  --seed data/sample/benchmark/seeds/s20260321 \
  --personas data/sample/personas_one.txt \
  --questions data/sample/questions_one.txt \
  --configs-root configs/few_shot \
  --output-dir /tmp/s2a_sample_few_shot_live \
  --allow-api-call

# Re-run extraction on one full-data persona (small API spend, one persona-level
# batch item; requires `make fetch` first)
printf 'bench_shift_121_avery_ellis\n' > /tmp/s2a_one_persona.txt
python -m survey2agent run-extraction \
  --provider openai --model gpt-5.4 \
  --seed data/benchmark/seeds/s20260321 \
  --personas /tmp/s2a_one_persona.txt \
  --output-dir /tmp/s2a_extraction \
  --allow-api-call

# Re-run few-shot on one full-data persona-question subset (small API spend)
PYTHONPATH=src python -m survey2agent run-few-shot \
  --provider openai --model gpt-5.4 \
  --seed data/benchmark/seeds/s20260321 \
  --personas /tmp/s2a_one_persona.txt \
  --questions A1,A2 \
  --configs-root configs/few_shot \
  --output-dir /tmp/s2a_few_shot \
  --allow-api-call
```

### Pipeline guarantees

- **Determinism**: All non-LLM components are deterministic given the seed. Any deviation between regenerated and released artifacts is a reproducibility bug — please file an issue.
- **Frozen LLM outputs**: The per-persona LLM outputs that underlie every paper number are shipped verbatim under `data/extracted_atoms/` and `data/method_outputs/`. See [CACHE_POLICY.md](CACHE_POLICY.md) for the frozen-vs-live layer split.
- **Live API calls are opt-in**: Re-running a fresh API call requires setting a provider API key in the environment and calling the relevant script directly. The default reproduction path never contacts any vendor API.
- **Batch-provider cache-only behavior**: For OpenAI and Gemini producer CLIs,
  cache-only mode is a safety boundary, not a replay path for
  `data/method_outputs/`. If a batch request is not already represented in the
  SHA256 cache, the command exits without network access unless
  `--allow-api-call` is supplied. Use `paper_artifacts/` to replay the frozen
  outputs used in the paper.

### Test suite on a fresh clone (`pytest`)

The repository ships without the expanded reproducibility bundle, so the data-touching tests are gated behind a `needs_data` pytest marker (registered in `pyproject.toml`, enforced by `tests/conftest.py`). When `data/benchmark/`, `data/extracted_atoms/`, and `data/method_outputs/` are not yet populated, the marker hook auto-skips every data-requiring test with the message:

> `requires data — run 'make fetch' or set S2A_DATA_ROOT to a populated bundle`

So a fresh `pytest` produces a clean `passed / skipped` summary instead of a `FileNotFoundError` flood. After `make fetch` (or with `S2A_DATA_ROOT` pointing at an existing bundle) the marker hook is a no-op and every `needs_data` test runs normally. To restrict collection to the data-bound subset, use `pytest -m needs_data`; to exclude it, use `pytest -m "not needs_data"`.

Static lint/type checks (`ruff`, `mypy`) are useful for development but are
not release gates for the anonymous artifact; the release gate is install,
fetch, smoke, paper-lock reproduction, aggregate regeneration, JSON parsing,
and the pytest suite.

---

## Open issues that affect reproducibility

(Updated continuously as discovered during Phase 6.)

- *(none flagged at Phase 1 scaffolding)*
