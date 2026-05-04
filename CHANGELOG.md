# Changelog

All notable changes to this project will be documented here. The format
loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning is per release tag.

## Unreleased

### Added
- `paper_artifacts/appendix/cross_bias_transfer.py`: cross-parameter
  transfer-without-refit reproduction for DSNBF (9 b×d variants × 3
  metrics = 27 paper-locked cells; ~9 min runtime). Reproduces the
  table in the paper Appendix robustness section subsection
  "Cross-Parameter Transfer Without Refit". Registered in
  `reproduce_paper._REGISTRY["appendix"]`, listed in `MANIFEST.md`,
  and gated by a `pytest.mark.slow` regression test
  (`test_paper_appendix_tier_b_reproduction.py::test_cross_bias_transfer_paper_lock`).
- Tier B reusability + extensibility surface:
  - `CONTRIBUTING.md`, `EXTENDING.md`, `SUBMISSION_PROTOCOL.md`,
    `ANONYMIZATION.md`
  - `examples/01_minimal_method/`, `examples/02_custom_question/`,
    `examples/03_programmatic_api/`
  - `schemas/method_prediction.schema.json`,
    `schemas/question_definition.schema.json`
  - `scripts/anonymization_audit.py` + CI integration
  - `paper_artifacts/verify_external_submission.py` (leaderboard verifier)
  - `paper_artifacts/leaderboard.csv` (stub with header)
  - Project-tracker templates (issue templates and PR template,
    distributed alongside the public mirror)
  - `Makefile` `seed-%` parametric target for re-generating one seed

### Changed
- CI workflow enabled anonymization audit + end-to-end smoke step;
  added `setuptools wheel` to the install line.
- `pyproject.toml`: added `jsonschema>=4.0` to the `dev` extra (used by
  the leaderboard verifier).

## 0.1.0 — initial release (2026-04-XX)

### Added
- Full method matrix — 12 deployable methods (T0 trivial through T3 LLM) plus the Source Reachability reference:
  Random, MajorityClass, SSB, SSB+SKIP, MajorityVote, BCF(4p), NBF,
  NBF+SKIP, DSNBF, DSNBF+SKIP, ABF, ABF+SKIP, OracleExtraction (Source Reachability reference, not a deployable method), plus
  LLM-Direct / LLM+SKIP / LLM-SchemaAware / LLM-FewShot variants.
- 4-seed evaluation (s20260321 / s20260322 / s20260323 / s20260324),
  pool-then-bootstrap 95% percentile CIs (B=2000, seed=42, persona-clustered).
- Paper-lock harness (`paper_artifacts/reproduce_paper.py`) regenerates
  23 tables (4 main + 19 appendix) from cached extractions.
- 480 personas / seed × 18 questions × 8 reasoning families benchmark.
