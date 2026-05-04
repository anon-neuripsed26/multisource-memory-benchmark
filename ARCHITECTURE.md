# Architecture Overview

This document describes what is in the release, what is not, and why.

---

## In scope (this repository)

The release contains the minimum code, data, and metadata required to:

1. Reproduce every numerical claim in the paper from cached API outputs (no API calls)
2. Re-run the deterministic pipeline (data generation, fusion methods, evaluation) end-to-end
3. Re-run a small slice of the API pipeline (extraction + LLM baselines) on a single persona for pipeline-level verification

The high-level layout is described in [README.md](README.md). The directory structure mirrors the responsibility split:

| Directory | Responsibility |
|-----------|----------------|
| `src/survey2agent/api_clients/` | `SyncLLMClient` (single-call, OpenRouter) and `BatchLLMClient` (provider batch APIs, OpenAI / Gemini) with SHA256 disk cache (forward-looking live-call layer; paper reproduction reads frozen `data/` instead) |
| `src/survey2agent/data_generation/` | Persona, event table, 5-source projection, ground-truth, NL render |
| `src/survey2agent/extraction/` | LLM-based atom extraction (`μ̂`) |
| `src/survey2agent/methods/` | Deployable methods plus the Source Reachability reference, spanning T0 trivial (Random, MajorityClass), T1 single-source (SSB, SSB+SKIP), T2 fusion (Majority-Vote/ArgRAG, BCF, NBF(+SKIP), DSNBF(+SKIP), ABF(+SKIP)), T3 LLM (LLM-Direct(+SKIP), Schema-Aware), and the Source Reachability reference (`OracleExtraction`, which uses GT only to probe direct readout from the structured source streams; not a deployable method). Auxiliary conditions (the matched-input GPT condition `StructLLMSource`, Few-Shot k=3) live alongside for the matched-input 2×2 factorial decomposition (figure in Experiments section) and the few-shot supplementary check (subsection of Selective QA Details). |
| `src/survey2agent/selective/` | SKIP calibration (θ_E, θ_Δ) and selective metrics |
| `src/survey2agent/evaluation/` | Accuracy, F0.5, bootstrap CI |
| `configs/` | Single source of truth for model IDs, question spec, seeds, splits |
| `scripts/` | Cross-platform Python entry points (NOT shell-first) |
| `paper_artifacts/` | One script per paper table / figure |
| `results/released/` | Fitted method parameters and final tables |
| `data/extracted_atoms/`, `data/method_outputs/` | Frozen per-persona LLM outputs that reproduce every paper number without API calls |
| `data/` | Small sample + Hugging Face fetch script for the full benchmark |
| `tests/` | Unit tests + one end-to-end smoke test (no API key required) |

---

## Out of scope (NOT in this release directory)

All release artifacts live under the repository root. Exploratory drafts, debugging scripts, alternative method prototypes, earlier benchmark revisions, and historical control-plane checkpoints are preserved on dedicated `archive/*` branches in repository history but are not shipped with the release. The `data/benchmarks/` legacy duplicate (≈2.4 GB) is excluded from the release; the 18-question benchmark used by the paper is fetched on demand by `data/fetch_benchmark.py`.

---

## Continuous integration

The default CI workflow covers deterministic generation, fusion methods, and evaluation only. API-backed tests (extraction, LLM baselines, cost pilot) require API keys and are gated behind a manually-triggered workflow that is **never** run automatically. This keeps:

- API secrets out of the default test surface
- CI reliable across forks and reviewers
- Spend predictable

---

## Versioning and reproducibility

The release is tagged `v0.1-neurips-submission`. Every commit on the main branch must keep `pytest` green. The frozen LLM outputs under `data/extracted_atoms/` and `data/method_outputs/` are the canonical source for paper reproduction; `python -m paper_artifacts.reproduce_paper` reads them directly and makes zero API calls. See [CACHE_POLICY.md](CACHE_POLICY.md) for the frozen-vs-live layer split and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full ML reproducibility checklist.
