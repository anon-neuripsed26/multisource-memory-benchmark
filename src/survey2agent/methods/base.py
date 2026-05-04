"""Method base class, prediction dataclass, and shared sentinels.

Every method (T0 trivial → T4 oracle) inherits from `Method` and implements
`predict_one(atom, qid) -> Prediction`. `fit` and `calibrate` are no-op by
default; subclasses override them iff `requires_fit` / `requires_calibration`
is set.

The `SKIP_SENTINEL` is the single canonical string for abstention. An
import-time guard checks that no question's `answer_space` contains the
sentinel, so SKIP can never collide with a real label.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS

SKIP_SENTINEL: str = "SKIP"

# Validate at import that no answer_space collides with SKIP_SENTINEL.
for _qid, _q in QUESTIONS.items():
    if SKIP_SENTINEL in _q["answer_space"]:
        raise RuntimeError(
            f"answer_space for {_qid} contains reserved sentinel {SKIP_SENTINEL!r}"
        )

GroundTruth = dict[str, str]


@dataclass(frozen=True)
class MethodTrainingRecord:
    """Per-persona fit/calibration record passed from the evaluator to methods.

    ``difficulty_class`` is metadata from the persona spec, not a prediction
    input for the target persona. Tuple-unpacking compatibility keeps the
    legacy ``for atom, gt in records`` method implementations working.
    """

    atom: ExtractedAtom
    gt: GroundTruth
    difficulty_class: str | None = None

    def __iter__(self):
        yield self.atom
        yield self.gt


TrainingRecord = MethodTrainingRecord | tuple[ExtractedAtom, GroundTruth]


@dataclass(frozen=True)
class Prediction:
    """One method's prediction for a single (persona, qid) pair.

    `would_skip` must be true iff `answer == SKIP_SENTINEL`. The
    runner aggregates `{qid: Prediction}` per persona into a single
    on-disk `MethodPrediction`.

    `raw_answer` is an optional pre-skip label used by LLM methods so the
    forced-mode evaluator can score the model's underlying choice even
    when the selective layer turned the prediction into a SKIP. Default
    `None` means the method does not distinguish raw from final (true of
    every non-LLM method).
    """

    answer: str
    would_skip: bool
    raw_answer: str | None = None

    def __post_init__(self) -> None:
        if (self.answer == SKIP_SENTINEL) != self.would_skip:
            raise ValueError(
                f"Prediction inconsistency: answer={self.answer!r}, would_skip={self.would_skip}"
            )
        if self.raw_answer == SKIP_SENTINEL:
            raise ValueError(
                f"raw_answer must not be SKIP_SENTINEL; use would_skip+answer instead"
            )
        if (
            not self.would_skip
            and self.raw_answer is not None
            and self.raw_answer != self.answer
        ):
            raise ValueError(
                f"For non-skip predictions, raw_answer must equal answer or be None "
                f"(answer={self.answer!r}, raw_answer={self.raw_answer!r})"
            )


class Method(ABC):
    """Abstract base for all methods.

    Subclasses set `name` (paper-facing identifier) and the two capability
    flags. The runner uses `requires_fit` / `requires_calibration` to
    decide whether to call `fit()` / `calibrate()` on the train / cal
    splits before evaluating on test.
    """

    name: str = ""
    requires_fit: bool = False
    requires_calibration: bool = False

    def fit(self, records: Sequence[TrainingRecord]) -> None:
        """Default no-op. Override iff `requires_fit = True`."""
        return None

    def calibrate(self, records: Sequence[TrainingRecord]) -> None:
        """Default no-op. Override iff `requires_calibration = True`."""
        return None

    @abstractmethod
    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        """Return a `Prediction` for one question of one persona."""

    def state_dict(self) -> dict:
        """Serialize learned parameters. Default: empty."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Inverse of `state_dict`. Default: no-op."""
        return None
