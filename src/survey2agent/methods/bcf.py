"""BCF: Bias-Corrected Fusion (paper Table 4, 4-parameter variant).

Treats every candidate answer as a hypothesis, projects it forward through
each active source's bias to get an "expected reading", and scores the
hypothesis by the (de-weighted) match count

    A_δ(v) = Σ_{s ∈ active} (1 - δ_s) · 𝟙[μ(s,Q) = bias_predict(s, Q, v)]

with `δ_obj = 0` (fixed objective-log anchor) and four learned deflations
``δ_prof, δ_plan, δ_self, δ_dev ∈ {0.0, 0.1, ..., 0.5}`` chosen by exhaustive
grid search on the train split (objective: training accuracy with
proportional credit for tied winners).

Byte-equivalent to the legacy v1.0 reference;
the 0-parameter ``BCF`` variant from legacy is intentionally not migrated
(only the 4-parameter version is reported in the paper).
"""

from __future__ import annotations

import itertools
import random
from typing import Sequence

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS, SOURCE_NAMES

from ._atom_adapter import atom_to_mu_q
from ._bias_model import bias_predict
from .base import Method, Prediction, TrainingRecord

# Sources with learnable deflation; objective_log is the fixed anchor (δ=0).
_LEARNABLE_SOURCES: tuple[str, ...] = (
    "profile_ltm",
    "planner",
    "daily_self_report",
    "device_log",
)
_GRID_STEPS: int = 6  # δ ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}


def _atom_to_mu_all(atom: ExtractedAtom) -> dict[str, dict[str, str | None]]:
    """Materialise per-question mu_q for every qid (used during fit)."""
    return {qid: atom_to_mu_q(atom, qid) for qid in QUESTIONS}


def _records_to_instances(records: Sequence[TrainingRecord]) -> list[dict]:
    """Project (ExtractedAtom, gt) records into legacy-shaped fit dicts.

    Inlined (rather than shared) to protect DSNBF byte-equivalence: see
    ``dsnbf.py`` for the parallel definition.
    """
    out: list[dict] = []
    for atom, gt in records:
        out.append({
            "persona": atom.persona,
            "gt": dict(gt),
            "mu": _atom_to_mu_all(atom),
        })
    return out


def _score_candidate(
    qid: str,
    cand: str,
    active: dict[str, str],
    deltas: dict[str, float],
) -> float:
    """A(v) = Σ_{s ∈ active} (1 - δ_s) · 𝟙[μ(s,Q) = bias_predict(s, Q, v)]."""
    score = 0.0
    for s, obs in active.items():
        if obs == bias_predict(s, qid, cand):
            score += 1.0 - deltas.get(s, 0.0)
    return score


class BCF(Method):
    """Bias-Corrected Fusion with 4 learned per-source deflation weights.

    Coverage = 100% (no SKIP). The reported `name` is ``"BCF(4p)"`` to match
    the paper's Table 4 / Table 5 column header.
    """

    name = "BCF(4p)"
    requires_fit = True
    requires_calibration = False

    def __init__(self, seed: int = 42) -> None:
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        self._deltas: dict[str, float] = {s: 0.0 for s in SOURCE_NAMES}

    # ---- fit ----

    def fit(self, records: Sequence[TrainingRecord]) -> None:
        instances = _records_to_instances(records)

        # Pre-compute per-(instance, qid) match matrix so the grid search
        # avoids redundant `bias_predict` calls. records_pre[i] =
        # (qid, gt, [(source, [match_cand_0, match_cand_1, ...]), ...])
        records_pre: list[tuple[str, str, list[tuple[str, list[bool]]]]] = []
        for inst in instances:
            for qid in QUESTIONS:
                mu_q = inst["mu"][qid]
                gt = inst["gt"][qid]
                active = {s: v for s, v in mu_q.items() if v is not None}
                if not active:
                    continue
                candidates = QUESTIONS[qid]["answer_space"]
                source_matches: list[tuple[str, list[bool]]] = []
                for s, obs in active.items():
                    matches = [obs == bias_predict(s, qid, c) for c in candidates]
                    source_matches.append((s, matches))
                records_pre.append((qid, gt, source_matches))

        cand_lists = {qid: QUESTIONS[qid]["answer_space"] for qid in QUESTIONS}
        grid_values = [i * 0.1 for i in range(_GRID_STEPS)]

        best_correct = -1.0
        best_deltas: dict[str, float] = {s: 0.0 for s in SOURCE_NAMES}

        for combo in itertools.product(grid_values, repeat=len(_LEARNABLE_SOURCES)):
            deltas = {s: d for s, d in zip(_LEARNABLE_SOURCES, combo)}
            deltas["objective_log"] = 0.0

            correct = 0.0
            for qid, gt, source_matches in records_pre:
                candidates = cand_lists[qid]
                n_cands = len(candidates)
                scores = [0.0] * n_cands
                for s, matches in source_matches:
                    w = 1.0 - deltas.get(s, 0.0)
                    for ci in range(n_cands):
                        if matches[ci]:
                            scores[ci] += w

                best_s = max(scores)
                winners = [ci for ci in range(n_cands) if scores[ci] == best_s]
                gt_idx = candidates.index(gt) if gt in candidates else -1
                if gt_idx in winners:
                    correct += 1.0 / len(winners)

            if correct > best_correct:
                best_correct = correct
                best_deltas = dict(deltas)

        self._deltas = best_deltas

    # ---- predict ----

    def _predict_label(self, qid: str, mu_q: dict[str, str | None]) -> str:
        candidates = QUESTIONS[qid]["answer_space"]
        active = {s: v for s, v in mu_q.items() if v is not None}

        if not active:
            return self._rng.choice(candidates)

        best_score = -1.0
        best_candidates: list[str] = []
        for cand in candidates:
            score = _score_candidate(qid, cand, active, self._deltas)
            if score > best_score:
                best_score = score
                best_candidates = [cand]
            elif score == best_score:
                best_candidates.append(cand)

        if len(best_candidates) == 1:
            return best_candidates[0]
        return self._rng.choice(sorted(best_candidates))

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        mu_q = atom_to_mu_q(atom, qid)
        return Prediction(answer=self._predict_label(qid, mu_q), would_skip=False)

    # ---- state persistence ----

    def state_dict(self) -> dict:
        return {
            "seed": self._seed,
            "deltas": dict(self._deltas),
        }

    def load_state_dict(self, state: dict) -> None:
        self._seed = int(state.get("seed", self._seed))
        self._rng = random.Random(self._seed)
        deltas = state.get("deltas", {})
        self._deltas = {s: float(deltas.get(s, 0.0)) for s in SOURCE_NAMES}
