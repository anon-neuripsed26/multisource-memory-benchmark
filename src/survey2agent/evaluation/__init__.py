"""Evaluation. Accuracy, F0.5, bootstrap CI, per-question
and per-difficulty aggregators.
"""

from .aggregator import aggregate_metrics
from .bootstrap import bootstrap_ci
from .breakdowns import breakdown_by, breakdown_by_qid_difficulty
from .data_loaders import (
    PROJECT_ROOT,
    TrainingRecord,
    build_training_records,
    load_atoms_for_seed,
    load_ground_truths,
    load_splits,
)
from .multi_seed import (
    CANONICAL_SEEDS,
    aggregate_per_seed_then_average,
    aggregate_per_seed_then_average_with_breakdown,
    aggregate_pooled_with_breakdown_ci,
    aggregate_pooled_with_ci,
    aggregate_pooled_with_qid_difficulty_ci,
    evaluate_method_across_seeds,
    pool_results_across_seeds,
)
from .runner import EvaluationResult, run_method

__all__ = [
    "PROJECT_ROOT",
    "TrainingRecord",
    "EvaluationResult",
    "aggregate_metrics",
    "bootstrap_ci",
    "breakdown_by",
    "breakdown_by_qid_difficulty",
    "build_training_records",
    "load_atoms_for_seed",
    "load_ground_truths",
    "load_splits",
    "run_method",
    "CANONICAL_SEEDS",
    "aggregate_per_seed_then_average",
    "aggregate_per_seed_then_average_with_breakdown",
    "aggregate_pooled_with_breakdown_ci",
    "aggregate_pooled_with_ci",
    "aggregate_pooled_with_qid_difficulty_ci",
    "evaluate_method_across_seeds",
    "pool_results_across_seeds",
]
