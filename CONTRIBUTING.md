# Contributing

Thanks for your interest in extending this benchmark. The repo is the
companion code & data for a NeurIPS 2026 Evaluations & Datasets Track
submission. The community goal is a long-lived, reusable testbed for
selective QA over multi-source personal memory.

This document covers governance, environment, code style, tests, and
the PR checklist. For *what* to extend (new method / question / seed /
stream / LLM backend), see [EXTENDING.md](EXTENDING.md).

## Code of conduct

This project follows the
[NeurIPS Code of Conduct](https://neurips.cc/public/CodeOfConduct).
Be kind, assume good faith, and prefer concrete reproducible artifacts
over opinion.

## Quickstart for contributors

```bash
# Clone the anonymous code mirror, then enter the repository.
git clone https://github.com/anon-neuripsed26/multisource-memory-benchmark.git
cd multisource-memory-benchmark
make install      # upgrades pip + installs editable package + dev extras
make smoke        # ~30 s end-to-end pipeline test
```

If `make smoke` is green you are ready to contribute. The full
reproduction (`make all` ≈ 25 min) is only needed when you change
something that affects paper tables.

## Reporting issues

Open an issue against the project tracker using one of the following
structured titles and bodies:

- **`[bug] <short title>`** — incorrect outputs, broken commands,
  broken reproduction, regressions in `make smoke` / `make reproduce`.
  Include: command run, expected vs actual output, environment
  (Python version + OS).
- **`[new method] <method-name>`** — propose adding a new method to
  the comparison matrix (please follow
  [EXTENDING.md §1](EXTENDING.md#1-add-a-new-method) before opening).
- **`[leaderboard] <method-name>`** — submit a new method's
  predictions for inclusion in the rolling leaderboard. See
  [SUBMISSION_PROTOCOL.md](SUBMISSION_PROTOCOL.md) for the required
  artifact format.

For ambiguous cases (is this a bug or expected behavior?), prefer a
small reproducer command that can be pasted into a fresh shell after
`make install`.

## Submitting a pull request

1. Branch from `main`. Use a short descriptive name
   (`feat/<short-name>`, `fix/<short-name>`, `docs/<short-name>`).
2. Keep the diff focused. Mixed PRs (e.g. "new method + reformat
   unrelated module") are hard to review.
3. Run the relevant test subset locally before pushing:
   ```bash
   pytest tests/methods/test_<your_change>.py -v   # focused
   make smoke                                       # always
   make test                                        # before merge
   ```
4. If your change affects any paper-locked claim, also re-run
   `make reproduce-main` (or `make reproduce-appendix` for appendix
   tables) and confirm the `paper_match` column stays `OK` for all
   cells. If a number must legitimately move, document that move in the
   PR body and update the corresponding paper-lock constant in the same
   PR.
5. Update `CHANGELOG.md` with a one-line entry under "Unreleased".
6. The CI workflow installs the package
   with `pip install -e .[dev]`, runs the deterministic pytest suite
   (`pytest tests -v -m "not requires_api"`), executes the smoke test
   (`pytest tests/test_smoke_end_to_end.py`), and runs
   `scripts/anonymization_audit.py` in warn-only mode during the
   double-blind period. CI does NOT install the `[api]` extra and does
   NOT exercise `make fetch`; tests that require the live benchmark
   directory rely on fixtures under `tests/`. Until camera-ready the
   audit must stay clean (zero findings).

## Code style and conventions

- **Python ≥ 3.10** (uses PEP 604 union types, `from __future__ import
  annotations`).
- **`pathlib.Path` everywhere**. No `os.path.join` in new code.
- **Methods inherit from `survey2agent.methods.base.Method`** and
  return `Prediction` objects. The `(answer == "SKIP") == would_skip`
  invariant is enforced in `Prediction.__post_init__`; do not bypass it.
- **One concept = one canonical predicate name** end-to-end. If you
  introduce a new predicate, name it once in the question spec / atom
  dataclass and reuse that exact name everywhere.
- **No GT or hidden-state leak into runtime.** Extraction must use only
  the visible source streams. Methods that train read GT only via the
  `fit` / `calibrate` callbacks, never inside `predict_one`.
- **No question-id or template-name routing inside methods.** A method
  must work uniformly across all 18 questions; per-question logic
  belongs in the data layer, not the method layer.
- **No hardcoded persona ids.** Use the persona iterator helpers in
  `survey2agent.extraction`.
- **Cache discipline**: cache keys are `(provider, api_model_id,
  prompt SHA-256)`. Do not cache by paper alias; do not write
  through the cache for live calls without setting `--allow-api-call`.
  See [CACHE_POLICY.md](CACHE_POLICY.md).

## Test discipline

- **Every new method ships with a test.** Minimum: a conformance test
  that instantiates the method and asserts every `predict_one` return
  is a valid `Prediction` for every question id in the fixture.
- **Tests must be deterministic.** Seed any RNG. The smoke test runs
  in CI; if your test takes more than 5 s, mark it `@pytest.mark.slow`
  so it stays out of the smoke set.
- **Fixture data lives in `tests/fixtures/`.** Do not load from
  `data/benchmark/...` inside unit tests; the smoke test is the only
  thing that touches the real benchmark.
- **Reproducibility tests** (`paper_artifacts/...`) must compare to the
  paper-lock constants at ±0.005 absolute. Loosen this only with
  explicit justification in the PR body.

## Releasing a new benchmark seed

When a new seed is generated and validated (see
[EXTENDING.md §3](EXTENDING.md#3-add-a-new-persona-generator-seed)):

1. Add the entry to `configs/seeds.yaml`.
2. Re-run `make reproduce-main` to confirm 4-seed aggregating tables
   pick it up.
3. Upload the seed directory to Hugging Face under the same dataset
   record (versioned by seed sub-directory).
4. Re-upload the new seed to Hugging Face (which refreshes the
   auto-generated Croissant for the dataset) and verify the dataset's
   `/croissant` endpoint with the
   [Croissant validator](https://huggingface.co/spaces/JoaquinVanschoren/croissant-checker).
5. Bump the dataset version in `pyproject.toml` and tag the commit.

## Dataset and code license

By contributing, you agree that your contribution will be released
under the project's existing licenses:

- Code: [Apache-2.0](LICENSE)
- Data: [CC-BY-4.0](DATA_LICENSE)

## Anonymization (during double-blind review)

Until camera-ready, the project is anonymized. PRs and issues that
inadvertently include identifying information (real name, real email,
ORCID, personal GitHub URL) will be edited or asked to be edited. See
[ANONYMIZATION.md](ANONYMIZATION.md) for the audit policy and
camera-ready reversal checklist.

## Questions

If something in this document is unclear, please open an issue with the
`docs` label rather than guessing — clarity gaps are bugs in this
document.
