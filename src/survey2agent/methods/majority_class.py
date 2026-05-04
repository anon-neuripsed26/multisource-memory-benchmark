"""T0 trivial baseline: per-question majority label learned on the train split."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS

from .base import Method, Prediction, TrainingRecord


class MajorityClass(Method):
    """Always predict the train-mode label per question.

    Tie-break: alphabetically first label among the tied set (deterministic).
    """

    name = "MajorityClass"
    requires_fit = True
    requires_calibration = False

    def __init__(self) -> None:
        self._mode: dict[str, str] = {}

    def fit(self, records: Sequence[TrainingRecord]) -> None:
        counters: dict[str, Counter] = {qid: Counter() for qid in QUESTIONS}
        for _atom, gt in records:
            for qid, gt_ans in gt.items():
                if qid in counters:
                    counters[qid][gt_ans] += 1
        self._mode = {}
        for qid, counter in counters.items():
            if not counter:
                # No train evidence: deterministic fallback to first label.
                self._mode[qid] = QUESTIONS[qid]["answer_space"][0]
                continue
            max_count = counter.most_common(1)[0][1]
            tied = sorted(label for label, cnt in counter.items() if cnt == max_count)
            self._mode[qid] = tied[0]

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        if qid not in self._mode:
            raise RuntimeError(
                f"MajorityClass.predict_one called before fit() (or qid {qid!r} missing)"
            )
        return Prediction(answer=self._mode[qid], would_skip=False)

    def state_dict(self) -> dict:
        return {"mode": dict(self._mode)}

    def load_state_dict(self, state: dict) -> None:
        self._mode = dict(state.get("mode", {}))
