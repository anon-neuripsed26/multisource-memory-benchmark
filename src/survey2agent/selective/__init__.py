"""Selective layer (macro-over-qid + micro siblings).

Provides:
- metrics: coverage, selective_accuracy / forced_accuracy / f_beta_selective
  (public dispatchers; macro when ``qids`` supplied, else micro+warn);
  ``*_macro`` and ``*_micro`` siblings; risk_coverage_curve, aurc.
- protocol: EvaluationOutcome enum, classify_prediction, reduce_outcomes
- thresholds: grid_search_f_beta (generic multi-axis threshold optimizer)

These are evaluation-only utilities. SKIP decision logic lives inside each
``*Selective`` method class (byte-equivalence lock with the published reference run).
"""

from .metrics import (
    aurc,
    coverage,
    f_beta_selective,
    f_beta_selective_macro,
    f_beta_selective_micro,
    forced_accuracy,
    forced_accuracy_macro,
    forced_accuracy_micro,
    risk_coverage_curve,
    selective_accuracy,
    selective_accuracy_macro,
    selective_accuracy_micro,
)
from .protocol import EvaluationOutcome, classify_prediction, reduce_outcomes
from .thresholds import grid_search_f_beta

__all__ = [
    "coverage",
    "selective_accuracy",
    "selective_accuracy_macro",
    "selective_accuracy_micro",
    "forced_accuracy",
    "forced_accuracy_macro",
    "forced_accuracy_micro",
    "f_beta_selective",
    "f_beta_selective_macro",
    "f_beta_selective_micro",
    "risk_coverage_curve",
    "aurc",
    "EvaluationOutcome",
    "classify_prediction",
    "reduce_outcomes",
    "grid_search_f_beta",
]
