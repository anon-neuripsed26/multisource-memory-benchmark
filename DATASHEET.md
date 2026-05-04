# Datasheet for the survey2agent Benchmark

This document follows the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) framework.

---

## Motivation

### For what purpose was the dataset created?

The dataset was created to support a diagnostic evaluation of selective question-answering methods over multi-source personal memory. Specifically, it provides a controlled testbed where:

- Personal memory is fully synthetic (no real-user data).
- Five evidence streams (`profile_ltm`, `planner`, `daily_self_report`, `objective_log`, `device_log`) project from a shared latent event table with **known, controlled per-source distortions** (bias direction, dropout rate, granularity).
- Ground-truth answers depend on the latent 30-day event history and the question; nine of the eighteen templates additionally read structured source annotations (documented in the paper appendix and in `data/README.md`). Ground-truth rules never read LLM-extracted atoms or any method prediction.
- Eighteen questions span eight reasoning types (Arbitration, Identity, Plan-Reality, Temporal-Trend, Factor-Attribution, Missing-Data, Annotation, Control).

### Who created the dataset?

(Anonymized for double-blind review.)

### Who funded the creation of the dataset?

(Anonymized for double-blind review.)

---

## Composition

### What do the instances represent?

Each instance is a `(persona, seed, question)` triple plus its deterministic ground-truth answer and the five evidence streams (in both structured and natural-language form).

### How many instances are there in total?

- 4 seeds × 480 personas × 18 questions = **34,560 instances**
- Per-seed: 480 personas × 18 questions = 8,640 instances
- Splits per seed: 216 train / 48 dev / 96 cal / 120 test (45 / 10 / 20 / 25 %)

### Does the dataset contain all possible instances or is it a sample?

It is a generated synthetic dataset. The persona generation script (`src/survey2agent/data_generation/generate_personas.py`) enumerates a deterministic set of `(seed, persona_id)` pairs. New seeds can be added without affecting existing instances.

### What data does each instance consist of?

For each persona (under `data/benchmark/seeds/<seed>/<persona_id>/`):
- `event_table.json` — 30-day latent event table (the "ground-truth world")
- `ground_truth.json` — 18 deterministic answers computed from `event_table.json` plus question-required structured source fields (9 of 18 templates read structured source annotations; the remaining 9 read only the latent event table)
- `structural_sources/profile_ltm.json` — long-term memory (stated profile; staleness / idealisation distortion)
- `structural_sources/planner.json` — planned activities (optimistic distortion vs habit parameters)
- `structural_sources/daily_self_report.json` — daily diary (topic-dependent self-report bias: Work −1, Diet +1, Social −1, Sleep +1, Exercise +1)
- `structural_sources/objective_log.json` — fitness/calendar log (small ± noise; most accurate)
- `structural_sources/device_log.json` — device data (precise where present; ~50% dropout on the work-session field plus day-level missingness)

The natural-language consolidation lives at the seed level, not the persona level: `data/benchmark/seeds/<seed>/nl_renders/<persona_id>.md` (used as input to LLM extraction and LLM-Direct). The same NL render is shared across all 4 seeds for a given persona slot; it is templated, not user-authored.

### Is there a label or target associated with each instance?

Yes. The `ground_truth.json` file gives the deterministic answer for each of the 18 questions, derived from the latent event table by a question-specific aggregation function `f_q`.

### Is any information missing from individual instances?

Sources contain spec-level missing-data patterns (e.g., `device_log` has ~50% dropout on the work-session field plus day-level missingness). This is a deliberate feature, not an annotation gap. Ground truth is always complete.

### Are relationships between individual instances made explicit?

Yes. Instances are organized by `(seed, persona_id)`. All 18 questions share the same five evidence streams within a persona.

### Are there recommended data splits?

Yes. See `configs/splits.yaml`. The split is per-persona within each seed: 216 train / 48 dev / 96 cal / 120 test. The split was chosen to give enough cal data for SKIP threshold calibration (96 personas × 18 questions = 1,728 calibration points per seed) while reserving a substantial test split.

### Are there any errors, sources of noise, or redundancies in the dataset?

The dataset is fully synthetic and has no human annotation step, so annotator-disagreement error does not apply. Label correctness still depends on the data-generating process and rule implementation. The deliberate distortions injected by the data-generation pipeline (per-source bias, dropout, idealization, granularity loss) are documented in paper Appendix B and `src/survey2agent/data_generation/`. Ground-truth correctness is verified by **deterministic re-execution**: the byte-equivalence smoke test [`tests/data_generation/test_smoke_byte_equivalence.py`](tests/data_generation/test_smoke_byte_equivalence.py) re-runs the full `L1 personas → L2 events → L3 sources → L4 ground_truth` pipeline from `seed=20260321` and asserts that the regenerated `ground_truth.json` files are byte-identical to the released artefact. The same holds for the other 3 seeds via the seed-parametric `make seed-%` target. No human annotation audit was performed; verification uses deterministic re-execution and an independent label-rule reimplementation with 100% label agreement, as reported in the paper appendix.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources?

The released dataset and the cached LLM outputs (`extracted_atoms/`, `method_outputs/`, `benchmark/results/`) are self-contained for reproducing every paper number with zero API calls (`make reproduce`). Live regeneration of LLM outputs (e.g. for a new model, prompt variant, or fifth seed) requires provider API keys and is gated behind the explicit `--allow-api-call` flag in the CLI runners.

### Does the dataset contain data that might be considered confidential or offensive?

No. All personas are synthetic. No real persons, locations, organizations, or events are referenced.

---

## Collection process

### How was the data associated with each instance acquired?

The data is generated by a deterministic Python pipeline (`src/survey2agent/data_generation/`) seeded by an integer:

1. Sample a persona prototype (demographics, habit parameters, difficulty class).
2. Roll a 30-day latent event table from habit parameters with the difficulty class controlling the realized-vs-stated gap.
3. Project five evidence streams from the latent event table according to per-source distortion specs.
4. Compute deterministic ground-truth answers from the latent event table and (for the 9 templates that require them) the corresponding structured source fields.
5. Render the five streams to natural language using a templated, non-LLM renderer.

### Over what time-frame was the data collected?

Generated in 2026. The benchmark is not time-locked; future seed runs are reproducible from the released code version (`v0.1-neurips-submission`) and the seed config in [`configs/seeds.yaml`](configs/seeds.yaml). Reproducibility is asserted only against this code version — running `generate_personas` from a different commit may produce a different persona slice if the data-generation pipeline has been changed.

### Seed naming convention

Each of the 4 seeds appears in three equivalent forms throughout the codebase and documentation. They all refer to the same RNG seed and the same 480-persona slice. The authoritative mapping lives in [`configs/seeds.yaml`](configs/seeds.yaml):

| Logical ID | Numeric (RNG seed) | String / directory ID | Role |
|------------|--------------------|------------------------|------|
| 1 | 20260321 | `s20260321` | development/supplementary seed; included in reported 4-seed pooled results |
| 2 | 20260322 | `s20260322` | evaluation seed; included in reported 4-seed pooled results |
| 3 | 20260323 | `s20260323` | evaluation seed; included in reported 4-seed pooled results |
| 4 | 20260324 | `s20260324` | evaluation seed; included in reported 4-seed pooled results |

- The **numeric form** (e.g. `20260321`) is the integer passed to `numpy.random.seed`. It encodes the date 2026-03-21..24, which is why the paper body uses the logical IDs `1, 2, 3, 4` rather than the numeric form (per the design rule in `configs/seeds.yaml`: *"Concrete numeric seeds are intentionally NOT shown in the paper body; they encode dates and appear only in code and appendix."*).
- The **string form** (`s20260321`, …) is the directory name under `data/benchmark/seeds/` and the canonical identifier passed to all Python APIs (e.g. `extraction.run_extraction(seed="s20260321")`).
- The **logical form** (`1, 2, 3, 4`) is the human-readable identifier used in paper body text ("seed 1" = dev = `s20260321`).

Aggregation conventions:
- **4-seed pooled**: instance pooled across all 4 seeds (480 × 4 = 1,920 personas), used by the main forced-accuracy table and by-reasoning-type table with persona-clustered bootstrap.
- **Per-seed-then-mean**: per-seed metric computed independently then averaged, used by the full selective-QA table because selective coverage drifts across seeds.
- **Single-seed (= seed 1 = `s20260321`)**: used by few-shot supplementary, prediction-distribution tables on failure questions, and GPT vs. Gemini cross-condition comparison.

### Persona Identifier Convention

Each persona has a string id of the form `bench_<prefix>_<NNN>_<name>` (e.g. `bench_shift_121_avery_ellis`). The `<prefix>` records the generation difficulty class for reporting and for reconstructing the persona spec:

| Prefix | Difficulty class |
|---|---|
| `stable` | `stable` |
| `shift` | `temporal_shift` |
| `stated` | `stated_vs_revealed` |

The canonical method that uses difficulty metadata during training is **DSNBF** (`src/survey2agent/methods/dsnbf.py`), which estimates separate emission tables from labeled training personas. In the official evaluation runner, train/calibration difficulty labels are passed from the persona spec metadata, not parsed from the target persona id. At prediction time, DSNBF receives only the observed atom table; it does not read the held-out `persona_id` or the true difficulty class, and instead infers a soft persona-local difficulty posterior from the observed source pattern. Other methods (Random, Majority-Class, SSB, Majority-Vote, BCF, NBF, ABF, LLM baselines, Oracle) are id-agnostic. External benchmark wrappers should strip or ignore identifiers before invoking prediction code; using the difficulty prefix as a feature is leakage and is outside the valid evaluation protocol.

For legacy direct tuple inputs that omit explicit difficulty metadata, DSNBF falls back to the historical id-prefix convention and assigns non-conforming ids to `stable` with a one-time `RuntimeWarning`; the global emission table still receives the persona's contribution, so fitting degrades gracefully rather than crashing. To exercise fully stratified DSNBF training on custom data, provide an equivalent training difficulty label.

### Were any ethical review processes conducted?

The dataset is fully synthetic and does not involve human subjects, so no IRB review was required.

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources?

Neither. The data is generated by a deterministic pipeline.

---

## Preprocessing / cleaning / labeling

### Was any preprocessing / cleaning / labeling of the data done?

The structured-source files are themselves outputs of the projection pipeline (no further preprocessing). Natural-language renders are generated from the structured files via a templated renderer (no LLM in the rendering loop). Ground truth is deterministic.

### Was the "raw" data saved in addition to the preprocessed / cleaned / labeled data?

The "raw" data is the latent event table (`event_table.json`), which is included with every persona. This allows researchers to re-derive any of the five sources or any new source projection.

### Is the software used to preprocess / clean / label the instances available?

Yes. See `src/survey2agent/data_generation/`.

---

## Uses

### Has the dataset been used for any tasks already?

The dataset was used to evaluate a deployable method suite (T0 trivial through T3 LLM) plus the Source Reachability reference (`OracleExtraction`, not deployable; uses GT only to probe direct readout from the structured source streams), with auxiliary conditions — the matched-input GPT condition (`StructLLMSource`, used in the 2×2 factorial decomposition) and the Few-Shot k=3 supplementary check — in the accompanying paper. See the paper's Experiments section and the Extended Diagnostic Tables and Robustness Analyses appendices.

### What other tasks could the dataset be used for?

- Bias-aware fusion algorithm research.
- Selective-QA / abstention method development.
- Benchmark for synthetic personal-memory models.
- Diagnostic stress test for LLM-based memory agents.

### Are there tasks for which the dataset should not be used?

- Studies that require real-user behavior data; this benchmark cannot substitute for that.
- Direct deployment to real users; the personas are explicitly synthetic.

---

## Distribution

### Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?

Yes. The dataset is hosted publicly on Hugging Face under CC BY 4.0.

### How will the dataset be distributed?

- Hugging Face Datasets (full)
- This repository's `data/sample/` (a small example)

### When will the dataset be distributed?

Anonymized release for review at submission time; public release at camera-ready.

### Will the dataset be distributed under a copyright or other intellectual-property (IP) license, and/or under applicable terms of use (ToU)?

The synthetic dataset itself — synthetic personas, generation code, ground-truth labels, schemas, and NL renders — is released under **CC BY 4.0**. See [DATA_LICENSE](DATA_LICENSE).

Cached LLM outputs in `benchmark/results/`, `extracted_atoms/`, and `method_outputs/` are responses from third-party model APIs (OpenAI / OpenRouter / Google Gemini / DeepSeek families) and are redistributed here only for byte-stable reproduction of the paper's tables. Downstream users who incorporate them into derivative work should follow the relevant provider terms of service; the per-provider links are listed in [`data/README.md`](data/README.md). The CC-BY-4.0 grant does not extend to those cached responses.

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

Yes, for cached LLM outputs only. The synthetic benchmark instances themselves
are not subject to third-party IP restrictions beyond the CC BY 4.0 terms above,
but frozen LLM responses in `benchmark/results/`, `extracted_atoms/`, and
`method_outputs/` remain subject to upstream provider terms and are included
only for reproducibility.

---

## Maintenance

### Who will be supporting / hosting / maintaining the dataset?

Hosted on Hugging Face. Maintained by the paper authors (anonymized for double-blind review). At camera-ready, this field will be replaced with the corresponding author's name and contact email per the de-anonymisation checklist in [ANONYMIZATION.md](ANONYMIZATION.md), and the dataset will be moved to a permanent organisation handle with a Zenodo DOI.

### How can the owner / curator / manager of the dataset be contacted?

Via the contact email listed in `CITATION.cff` (anonymized for review).

### Is there an erratum?

None at submission time. Errata, if any, will be posted in the repository's `CHANGELOG.md` and on the Hugging Face dataset page.

### Will the dataset be updated?

Possibly, to add additional seeds or new questions. Backwards compatibility will be preserved (existing seeds and questions will not be silently changed). Any change will be versioned and announced.

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?

The dataset does not relate to real people.

### Will older versions of the dataset continue to be supported / hosted / maintained?

Yes. Tagged versions (e.g., `v0.1-neurips-submission`) will remain accessible.

### If others want to extend / augment / build on / contribute to the dataset, is there a mechanism for them to do so?

Yes. The data-generation pipeline is open source (Apache 2.0). The five extension points — new method, new question, new evidence stream, new LLM backend, programmatic-API driver — are documented in [EXTENDING.md](EXTENDING.md) §1-§5 with runnable examples under `examples/01_minimal_method/`, `examples/02_custom_question/`, and `examples/03_programmatic_api/`. External method submissions to the rolling leaderboard follow the contract in [SUBMISSION_PROTOCOL.md](SUBMISSION_PROTOCOL.md).
