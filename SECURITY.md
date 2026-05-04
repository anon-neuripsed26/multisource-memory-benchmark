# Security Policy

## Scope

This repository contains a research benchmark and evaluation harness. It
does not host a production service, accept user uploads, or process
real personal data — every persona in `data/benchmark/` is fully
synthetic (see `DATASHEET.md`). The threat surface is therefore narrow:

- Code execution from cloning + running scripts/tests.
- Dependency supply chain (`pyproject.toml`).
- Optional outbound calls to LLM providers when the user opts in to
  the `[api]` extra and re-runs extraction (default reproduction is
  cache-only and makes zero network calls).

## Reporting a Vulnerability

During the NeurIPS 2026 double-blind review period, please file a
private message to the program chairs through OpenReview rather than
opening a public issue. Include:

1. Affected file path(s) and commit hash.
2. A short reproducer (script or commands).
3. Observed vs. expected behavior.

After the review period, the maintainer contact listed in `CITATION.cff`
will be the responsible party. We aim to acknowledge reports within 7
days and triage them within 30 days.

## Out of Scope

- Findings that depend on the user voluntarily executing untrusted code
  pulled from outside this repository.
- Issues in third-party model providers (OpenAI, Google, Anthropic,
  DeepSeek, Qwen, etc.) accessed through the optional `[api]` extra.
- Anonymization regressions: those are tracked under `ANONYMIZATION.md`
  and audited via `scripts/anonymization_audit.py`.
