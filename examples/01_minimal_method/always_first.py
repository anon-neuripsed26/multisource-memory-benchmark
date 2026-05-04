"""`always_first` — a 30-line minimal method.

Always picks the first label in each question's answer space. Useful as
a sanity baseline (does the framework wire up at all?) and as the
canonical "how to write a new method" example.

See ../README.md for the walkthrough this file accompanies.
"""

from __future__ import annotations

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS
from survey2agent.methods.base import Method, Prediction


class AlwaysFirst(Method):
    """Deterministic baseline: predict `answer_space[0]` for every question."""

    name = "AlwaysFirst"
    requires_fit = False
    requires_calibration = False

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        answer = QUESTIONS[qid]["answer_space"][0]
        return Prediction(answer=answer, would_skip=False)
