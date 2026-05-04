"""T0 trivial baseline: uniform random from a question's answer space."""

from __future__ import annotations

import random

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS

from .base import Method, Prediction


class Random(Method):
    """Sample a label uniformly at random from `QUESTIONS[qid].answer_space`."""

    name = "Random"
    requires_fit = False
    requires_calibration = False

    def __init__(self, seed: int | None = 42) -> None:
        # ``seed`` defaults to 42 (paper canonical) for deterministic output.
        self._rng = random.Random(seed)

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        answer = self._rng.choice(QUESTIONS[qid]["answer_space"])
        return Prediction(answer=answer, would_skip=False)
