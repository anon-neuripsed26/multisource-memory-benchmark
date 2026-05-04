"""Shared helpers for Tier C robustness appendix scripts.

Centralizes paths to ``$DATA_ROOT/benchmark/results`` JSON artifacts and
the naming-translation table from paper labels (DSNBF, NBF, ABF, BCF, MV,
SSB, MC) to JSON keys (DSNBF-NoSkip, NBF-NoSkip, PRISM-NoSkip, BCF(4p),
Majority-Vote, Single-Source-Best, Majority-Class).

Verified against ``$DATA_ROOT/benchmark/results/4seed_results.json`` and
``ablation_experiments_s2026032{1-4}.json``:

  ABF  -> PRISM-NoSkip   (4-seed mean 71.78%, paper Training Size Sensitivity 71.8±1.0)
  SSB  -> Single-Source-Best (4-seed mean 79.03%, paper Cross-Extractor Robustness 79.0)
  BCF  -> BCF(4p)        (4-seed mean 69.35%, paper Training Size Sensitivity 69.4±0.7)
  MV   -> Majority-Vote
  MC   -> Majority-Class

The legacy ``SSB-Global`` key (uniform global source-selection) is *not*
the same as paper SSB; we never use it for paper-lock.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from survey2agent._paths import RESULTS_ROOT

# Backward-compatible alias for the few callers that still import this name.
RESULTS_DATA_DIR: Path = RESULTS_ROOT

CANONICAL_SEED_INTS: tuple[int, ...] = (20260321, 20260322, 20260323, 20260324)
CANONICAL_SEED_STRS: tuple[str, ...] = tuple(f"s{s}" for s in CANONICAL_SEED_INTS)
CANONICAL_SEED_KEYS: tuple[str, ...] = tuple(str(s) for s in CANONICAL_SEED_INTS)


# Paper-label -> JSON-key mapping (see module docstring for verification)
PAPER_LABEL_TO_JSON_KEY: dict[str, str] = {
    "DSNBF":   "DSNBF-NoSkip",
    "NBF":     "NBF-NoSkip",
    "ABF":     "PRISM-NoSkip",
    "BCF":     "BCF(4p)",
    "MV":      "Majority-Vote",
    "SSB":     "Single-Source-Best",
    "MC":      "Majority-Class",
}


def load_4seed_results() -> dict:
    """Load ``data/benchmark/results/4seed_results.json`` (the canonical
    multi-seed fusion-method aggregator)."""
    return json.loads(
        (RESULTS_DATA_DIR / "4seed_results.json").read_text(encoding="utf-8")
    )


def load_ablation(seed: str) -> dict:
    """``ablation_experiments_<seed>.json`` -> {ablation: {N: {method:acc}},
    perturbed: {eps: {method:acc}}}. ``seed`` form: ``"s20260321"``."""
    return json.loads(
        (RESULTS_DATA_DIR / f"ablation_experiments_{seed}.json").read_text(
            encoding="utf-8"
        )
    )


def load_full_comparison(seed: str, *, gemini: bool = False) -> dict:
    """``full_comparison[_gemini]_<seed>.json``. Only the ``extraction``
    block is paper-relevant for Tier C (the ``llm`` block in the gemini
    file is a copy of GPT values, do not read)."""
    fname = f"full_comparison_{'gemini_' if gemini else ''}{seed}.json"
    return json.loads((RESULTS_DATA_DIR / fname).read_text(encoding="utf-8"))


def load_bootstrap_ci(*, all_methods_v2: bool = False) -> dict:
    """``bootstrap_ci_4seed.json`` (default) or
    ``bootstrap_ci_all_methods_v2.json`` (which adds Struct-* rows)."""
    fname = "bootstrap_ci_all_methods_v2.json" if all_methods_v2 else "bootstrap_ci_4seed.json"
    return json.loads((RESULTS_DATA_DIR / fname).read_text(encoding="utf-8"))


def load_openweight_4seed() -> dict:
    """``openweight_4seed_eval.json`` — DeepSeek + Qwen3 per-seed macros
    in *percent* (not fraction)."""
    return json.loads(
        (RESULTS_DATA_DIR / "openweight_4seed_eval.json").read_text(encoding="utf-8")
    )


def load_c6b_perq_perdiff_4seed() -> dict:
    """``c6b_perq_perdiff_4seed.json`` — keys ``Q__diff__ext`` ->
    ``{mean, std, per_seed, total_per_seed}``. ``mean`` is in percent."""
    return json.loads(
        (RESULTS_DATA_DIR / "c6b_perq_perdiff_4seed.json").read_text(encoding="utf-8")
    )


# ── Statistics helpers ────────────────────────────────────────────────


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Return (mean, population stddev) of a non-empty sequence.

    Population stddev (divisor N, not N-1) was empirically the closest
    match to paper Cross-Seed Stability σ values (DSNBF σ_pop=0.62 -> paper 0.6;
    sample σ would round to 0.7). Some paper tables (e.g. Training Size Sensitivity) use
    sample stddev instead — we expose population here and let the
    paper-lock tolerance (±0.002 absolute on σ) absorb the rounding
    inconsistency. See report for which tables show σ disagreement.
    """
    if not values:
        raise ValueError("mean_std requires non-empty sequence")
    return statistics.mean(values), statistics.pstdev(values)
