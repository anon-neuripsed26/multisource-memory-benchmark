"""Selective evaluation outcome semantics.

Per-prediction outcome labels for cost-asymmetric scoring. Maps each
``(Prediction, ground-truth label)`` pair to one of three categorical
outcomes used by `f_beta_selective` and downstream reporting.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Sequence

from survey2agent.methods.base import SKIP_SENTINEL, Prediction


class EvaluationOutcome(str, Enum):
    """Per-prediction outcome label.

    - ``TRUE_ANSWER``: ``would_skip=False`` and ``answer == label`` (TP).
    - ``FALSE_ANSWER``: ``would_skip=False`` and ``answer != label`` (FP).
    - ``SKIP``: ``would_skip=True`` (FN under the "skip is never GT" rule).
    """

    TRUE_ANSWER = "true_answer"
    FALSE_ANSWER = "false_answer"
    SKIP = "skip"


def classify_prediction(pred: Prediction, label: str) -> EvaluationOutcome:
    """Map a prediction and its ground-truth label to an `EvaluationOutcome`.

    Raises ``ValueError`` if ``label == SKIP_SENTINEL`` (GT cannot be SKIP).
    """
    if label == SKIP_SENTINEL:
        raise ValueError(
            f"ground-truth label must not be the SKIP sentinel ({SKIP_SENTINEL!r})"
        )
    if pred.would_skip:
        return EvaluationOutcome.SKIP
    if pred.answer == label:
        return EvaluationOutcome.TRUE_ANSWER
    return EvaluationOutcome.FALSE_ANSWER


def reduce_outcomes(outcomes: Sequence[EvaluationOutcome]) -> dict[str, int]:
    """Count each outcome type.

    Always returns all three keys (``true_answer``, ``false_answer``, ``skip``)
    with zero counts for absent categories.
    """
    counts = Counter(outcomes)
    return {
        EvaluationOutcome.TRUE_ANSWER.value: int(counts[EvaluationOutcome.TRUE_ANSWER]),
        EvaluationOutcome.FALSE_ANSWER.value: int(counts[EvaluationOutcome.FALSE_ANSWER]),
        EvaluationOutcome.SKIP.value: int(counts[EvaluationOutcome.SKIP]),
    }
