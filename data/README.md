---
license: cc-by-4.0
language:
- en
pretty_name: "Multi-Source Memory Benchmark (anonymous NeurIPS 2026 submission)"
size_categories:
- 10K<n<100K
task_categories:
- question-answering
- text-classification
tags:
- benchmark
- selective-qa
- personal-memory
- multi-source
- llm-evaluation
- conflict-resolution
- synthetic
configs:
- config_name: s20260321
  data_files:
  - split: train
    path: "benchmark/seeds/s20260321/config/persona_splits.json"
- config_name: s20260322
  data_files:
  - split: train
    path: "benchmark/seeds/s20260322/config/persona_splits.json"
- config_name: s20260323
  data_files:
  - split: train
    path: "benchmark/seeds/s20260323/config/persona_splits.json"
- config_name: s20260324
  data_files:
  - split: train
    path: "benchmark/seeds/s20260324/config/persona_splits.json"
---

# Multi-Source Memory Benchmark

> **Status — anonymous artefact for double-blind review (NeurIPS 2026 Evaluations & Datasets Track).**
> Author identities, organisations, and funders are intentionally withheld until the review period concludes.

A diagnostic testbed for **selective question-answering (`ANSWER` / `SKIP`) over conflicting multi-source personal memory.**
Each persona has five evidence streams projected from a single latent event table with **known, controlled per-source distortions** (bias direction, dropout rate, granularity), allowing methods to be measured against the *latent ground truth* rather than against any single source.

The benchmark accompanies the paper *"Selective QA over Conflicting Multi-Source Personal Memory: A Diagnostic Testbed and Method Comparison"* (anonymous, NeurIPS 2026 Evaluations & Datasets Track submission). It is one of two artefacts; the **code** mirror is hosted at <https://github.com/anon-neuripsed26/multisource-memory-benchmark> and linked from the paper's Reproducibility section.

The accompanying method comparison spans **baselines, structured fusion methods, and frontier LLMs** (GPT, Gemini, DeepSeek, Qwen3 families).

> **Hugging Face viewer note.** The Dataset Viewer shows one split-summary
> record per seed config, so the page may display `<1K` rows. The benchmark is
> file-based: the 34,560 question instances live under `benchmark/seeds/` and
> in the checksum-verified archive described below.

---

## Quick start

```bash
pip install huggingface_hub
# Code mirror: https://github.com/anon-neuripsed26/multisource-memory-benchmark
# (Clone it with:
#   git clone https://github.com/anon-neuripsed26/multisource-memory-benchmark.git
#   cd multisource-memory-benchmark)
python data/fetch_benchmark.py            # downloads a ~36 MB ZIP; expands to ~410 MB
make smoke                                # fast byte-equivalence check (~30 s)
```

The recommended fetch path downloads
`archives/multisource-memory-benchmark-data-v0.1.0.zip`, verifies its
SHA256 checksum, extracts `benchmark/`, `extracted_atoms/`, and
`method_outputs/`, then downloads the small top-level metadata files. This
avoids Hugging Face rate limits that can occur when fetching the expanded
~29k-file tree one file at a time. To force the expanded snapshot fallback,
use `S2A_FETCH_MODE=snapshot python data/fetch_benchmark.py`.

Or load a single persona programmatically:

```python
import json
from pathlib import Path
persona_dir = Path("data/benchmark/seeds/s20260321/bench_shift_001_drew_carter")
gt = json.loads((persona_dir / "ground_truth.json").read_text())
print(list(gt.keys())[:5])  # ['A1', 'A2', 'A3', 'B2', 'B3']
```

---

## Repository layout

```
.
├── benchmark/                       (~370 MB)
│   ├── seeds/
│   │   ├── s20260321/               # dev seed (480 personas)
│   │   │   ├── bench_shift_001_drew_carter/  # one persona = one folder
│   │   │   │   ├── event_table.json          # latent day-level world state
│   │   │   │   ├── ground_truth.json         # 18 deterministic answers
│   │   │   │   └── structural_sources/
│   │   │   │       ├── profile_ltm.json
│   │   │   │       ├── planner.json
│   │   │   │       ├── daily_self_report.json
│   │   │   │       ├── objective_log.json
│   │   │   │       └── device_log.json
│   │   │   ├── ... (480 personas) ...
│   │   │   ├── nl_renders/                   # NL-rendered memories (480 × .md)
│   │   │   └── config/
│   │   │       ├── personas.json
│   │   │       └── persona_splits.json       # train/dev/cal/test split
│   │   ├── s20260322/                # eval seeds (3 × 480 personas)
│   │   ├── s20260323/
│   │   └── s20260324/
│   └── results/                     # 32 per-method JSONs across 4 seeds
├── extracted_atoms/                 (~2 MB)   # frozen LLM-extracted atoms
├── method_outputs/                  (~30 MB)  # frozen per-method outputs
├── README.md                        # this file (HF dataset card)
├── DATA_LICENSE                     # full text of CC-BY-4.0
├── DATASHEET.md                     # Gebru et al. (2018) datasheet answers
├── CITATION.cff                     # citation entry (anonymised)
└── CROISSANT_RAI.json               # completed Croissant metadata for OpenReview
```

`CROISSANT_RAI.json` is the authoritative Croissant+RAI metadata file for
OpenReview. Hugging Face may also expose an auto-generated `/croissant` endpoint
or lightweight viewer representation; those platform-generated views are not
the artifact-level metadata used for review.

---

## Benchmark statistics

| Quantity | Value |
|---|---|
| Seeds | 4 (s20260321 used for development/supplementary checks; all 4 included in reported pooled results) |
| Personas per seed | 480 |
| Questions per persona | 18 (8 reasoning types) |
| Total instances | 4 × 480 × 18 = **34 560** |
| Difficulty classes | 3 (stable / temporal_shift / stated_vs_revealed), 160 personas each per seed |
| Per-seed split | 216 train / 48 dev / 96 cal / 120 test (45 / 10 / 20 / 25 %) |
| Reasoning types | A-Arbitration · B-Identity · C-Plan–Reality · D-Temporal-Trend · E-Factor · F-Missing-Data · G-Annotation · Ctrl-Control |
| Topics | Work · Diet · Social · Sleep · Exercise |
| Answer space | 15 ordinal questions, 3 nominal questions |
| Synthetic? | **Yes — 100 % synthetic.** No real-user data is included. |

Five evidence streams per persona, each with a *known* distortion profile:

| Stream | Distortion | Bias direction `b` | Notes |
|---|---|---|---|
| `profile_ltm` | Staleness / idealisation | ≈ 0 | Coarse prior; mixture absorbed by learned δ_prof |
| `planner` | Optimistic vs habit | +1 | Starts from habit parameters |
| `daily_self_report` | **Topic-dependent** | ±1 (Work −1, Diet +1, Social −1, Sleep +1, Exercise +1) | Most diverse |
| `objective_log` | Small ± noise | 0 | Most accurate; δ fixed at 0 |
| `device_log` | ~50 % dropout on the work-session field + day-level missingness | 0 | Precise where present |

Ground truth is computed deterministically from the latent persona state
and the question template. All labels depend on the latent 30-day event
history and the question; nine templates additionally read structured
source annotations, as documented in the paper appendix. The answer
rules do not read the LLM-extracted atoms or any method prediction.

---

## Intended use

This dataset is a **diagnostic benchmark**. It is intended for:

- Comparing selective-QA aggregation methods (single-source, fusion, end-to-end LLM, oracle) under controlled per-source distortion.
- Stress-testing how methods handle conflicting evidence, missing fields, and topic-dependent self-report bias.
- Studying the cost-of-skip vs cost-of-wrong trade-off in personal-memory QA.

It is **not** intended for:

- Training production personal-memory assistants on real users (the personas are synthetic and statistically simplified).
- Studying realistic free-text disclosure or privacy attacks (the natural-language renders are templated, not user-authored).
- Benchmarking general-purpose LLM reasoning outside the selective-QA framing (the question set is closed and small).

---

## Cached LLM outputs and provider terms

`benchmark/results/`, `extracted_atoms/`, and `method_outputs/` contain **cached outputs** from third-party model APIs (variants of GPT-5, Gemini 3, Qwen3, and DeepSeek-V3.2). They are released here **only for exact reproducibility of the paper's tables**. The shipped cache avoids additional reviewer API spend and keeps the reported numbers byte-stable. The `extracted_atoms/` cache covers the held-out test split; structured fusion fitting and calibration in the reproduction path use deterministic direct-readout atoms from the train/calibration splits, then evaluate the reported `\hat{\mu}` rows on these frozen test extractions.

These cached outputs are generated by third-party model APIs. Users who incorporate them into derivative work should follow the relevant provider terms:

- OpenAI: <https://openai.com/policies/terms-of-use>
- OpenRouter: <https://openrouter.ai/terms> (provider-specific terms apply per upstream model)
- Google Gemini: <https://ai.google.dev/gemini-api/terms>
- DeepSeek: <https://platform.deepseek.com/downloads/DeepSeek%20Open%20Platform%20Terms%20of%20Service.html> (note: derivative content must be labelled as AI-generated where required)

The CC-BY-4.0 license below covers the **dataset structure, the synthetic personas, and the benchmark schema**; cached model outputs are redistributed under the terms of the upstream providers and are flagged as such here.

Legacy result keys are confined to cached result JSONs: `PRISM` /
`PRISM-NoSkip` are the pre-submission keys for ABF / ABF-NoSkip. Paper-facing
tables, captions, and reproduction scripts map them deterministically to ABF.

---

## Limitations and bias

- **Synthetic, not field-collected.** Personas are sampled from coded distributions; real users will exhibit dependencies and rare events not modelled here.
- **Western-leaning template.** Activity types (gym, run, cardio, etc.), measurement units (calories, hours), and natural-language renders are in English with US-style conventions.
- **Topic coverage is narrow.** The 18 questions span 5 topics; broader life domains (finance, health conditions, relationships) are intentionally out of scope.
- **Optimised for selective-QA diagnosis.** The dataset is *not* a leaderboard for general LLM reasoning quality.

See [`DATASHEET.md`](../DATASHEET.md) for the full Datasheets-for-Datasets answers (composition, collection, preprocessing, uses, distribution, maintenance). The completed OpenReview Croissant+RAI submission file is [`CROISSANT_RAI.json`](CROISSANT_RAI.json); do not substitute the platform-generated Hugging Face viewer metadata for this file.

---

## License

The benchmark dataset (synthetic personas, generation code, ground-truth labels, schemas, NL renders) is released under **CC-BY-4.0** — see [`DATA_LICENSE`](DATA_LICENSE).

Cached LLM outputs in `benchmark/results/`, `extracted_atoms/`, and `method_outputs/` are governed by the upstream provider terms of service linked above.

---

## Citation

```bibtex
@misc{anonymous_2026_selective_qa_memory,
  title         = {Selective QA over Conflicting Multi-Source Personal Memory: A Diagnostic Testbed and Method Comparison},
  author        = {Anonymous Authors},
  year          = {2026},
  note          = {Anonymous submission, NeurIPS 2026 Evaluations \& Datasets Track. De-anonymised version will be released upon acceptance.}
}
```

A machine-readable `CITATION.cff` is included.

---

## Maintenance

This artefact will be replaced with a permanent, de-anonymised release at the project's maintainer organisation upon paper acceptance, with a Zenodo DOI for archival.

---

## For developers (working in the code repo)

This file doubles as the local-`data/`-directory README inside the code repository.
The contents under this directory are the runtime data root (`$S2A_DATA_ROOT`).
When `S2A_DATA_ROOT` is unset, the package defaults to `data/`
(see [`survey2agent/_paths.py`](../src/survey2agent/_paths.py)).

To redirect the entire data root (e.g. to a fast SSD or a separate mount):

```bash
export S2A_DATA_ROOT=/path/to/your/data
python data/fetch_benchmark.py
```

Then run the test suite to confirm everything resolves:

```bash
PYTHONPATH=src python3 -m pytest tests/ -q
```

### Regenerating `benchmark/seeds/` from source

If you want to reproduce `benchmark/seeds/` from the data-generation
pipeline rather than download it from Hugging Face:

```bash
python -m survey2agent.data_generation.generate_personas \
    --seed 20260321 --output-dir data/benchmark/seeds/s20260321
python -m survey2agent.data_generation.generate_events    --dataset-dir data/benchmark/seeds/s20260321
python -m survey2agent.data_generation.generate_sources   --dataset-dir data/benchmark/seeds/s20260321
python -m survey2agent.data_generation.generate_ground_truth --dataset-dir data/benchmark/seeds/s20260321
```

Both routes are byte-equivalent (verified by
`tests/data_generation/test_smoke_byte_equivalence.py`).
For a small end-to-end generation smoke outside the canonical release,
use `generate_personas --per-difficulty 10`. Extremely tiny settings
such as `--per-difficulty 1` may fail the diversity audit thresholds that
protect the released benchmark distribution.

### Path API

Code should never hard-code paths. Use the helpers in `survey2agent._paths`:

```python
from survey2agent._paths import (
    DATA_ROOT,             # = $S2A_DATA_ROOT (or default)
    EXTRACTED_ATOMS_ROOT,  # = $DATA_ROOT/extracted_atoms
    METHOD_OUTPUTS_ROOT,   # = $DATA_ROOT/method_outputs
    BENCHMARK_ROOT,        # = $DATA_ROOT/benchmark
    SEEDS_ROOT,            # = $DATA_ROOT/benchmark/seeds
    RESULTS_ROOT,          # = $DATA_ROOT/benchmark/results
    seed_dir,              # seed_dir("s20260321") → SEEDS_ROOT/s20260321
    persona_dir,           # persona_dir(seed, persona_id)
    nl_renders_dir,        # nl_renders_dir(seed)
)
```
