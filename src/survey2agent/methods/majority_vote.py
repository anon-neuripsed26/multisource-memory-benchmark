"""T2 stateless fusion baseline: plurality vote across non-null sources."""

from __future__ import annotations

import random
from collections import Counter

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS

from ._atom_adapter import atom_to_mu_q
from .base import Method, Prediction


class MajorityVote(Method):
    """Plurality vote on non-null source values; seeded random tie-break.

    All-null edge case: fall back to a uniform random label from the
    question's answer space (matches legacy behavior; no SKIP because
    this is the non-selective family).
    """

    name = "MajorityVote"
    requires_fit = False
    requires_calibration = False

    def __init__(self, seed: int | None = 42) -> None:
        # ``seed`` defaults to 42 (paper canonical) for deterministic
        # tie-breaking and null-source fallback.
        self._rng = random.Random(seed)

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        mu_q = atom_to_mu_q(atom, qid)
        votes = [v for v in mu_q.values() if v is not None]
        if not votes:
            return Prediction(
                answer=self._rng.choice(QUESTIONS[qid]["answer_space"]),
                would_skip=False,
            )
        counter = Counter(votes)
        max_count = counter.most_common(1)[0][1]
        tied = [label for label, cnt in counter.items() if cnt == max_count]
        if len(tied) == 1:
            return Prediction(answer=tied[0], would_skip=False)
        return Prediction(answer=self._rng.choice(sorted(tied)), would_skip=False)
