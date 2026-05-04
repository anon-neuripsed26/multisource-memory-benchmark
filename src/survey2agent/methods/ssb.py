"""T1 single-source baselines: SSB and SSB+SKIP (selective variant).

SSB picks, per question, the source whose extraction maximises expected
accuracy on the train split (with a random fallback when the source value
is `None`, identical to the runtime predict path).

`SSBSelective` extends SSB with two abstention rules:
  * C1: the chosen source is null for this persona/question -> SKIP
  * C2: among non-null sources, fewer than `theta_agree` agree with the
    chosen source's value -> SKIP. C2 only fires when at least 2 sources
    are active (single-active-source cases always answer).

`theta_agree` is selected on the cal split by grid-search to maximise the
F_{0.5} score (precision-weighted) of the answer/skip decision.
"""

from __future__ import annotations

import random
from typing import Sequence

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS, SOURCE_NAMES

from ._atom_adapter import atom_to_mu_q
from .base import SKIP_SENTINEL, Method, Prediction, TrainingRecord


def _fit_best_source_per_question(
    records: Sequence[TrainingRecord],
) -> dict[str, str]:
    """Return `{qid: best_source}` by maximising `correct + n_null / k`."""
    best: dict[str, str] = {}
    for qid in QUESTIONS:
        k = len(QUESTIONS[qid]["answer_space"])
        best_score = -1.0
        best_source = SOURCE_NAMES[0]
        for src in SOURCE_NAMES:
            correct = 0
            n_null = 0
            for atom, gt in records:
                if qid not in gt:
                    continue
                mu_q = atom_to_mu_q(atom, qid)
                val = mu_q[src]
                if val is None:
                    n_null += 1
                elif val == gt[qid]:
                    correct += 1
            score = correct + n_null / k
            if score > best_score:
                best_score = score
                best_source = src
        best[qid] = best_source
    return best


class SSB(Method):
    """Single-Source-Best: per-question best-source selection (no SKIP).

    ``seed`` defaults to 42 (paper canonical) for deterministic tie-breaking and
    null-source fallback. Pass an explicit value to override.
    """

    name = "SSB"
    requires_fit = True
    requires_calibration = False

    def __init__(self, seed: int | None = 42) -> None:
        self._best_source: dict[str, str] = {}
        self._rng = random.Random(seed)

    def fit(self, records: Sequence[TrainingRecord]) -> None:
        self._best_source = _fit_best_source_per_question(records)

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        if qid not in self._best_source:
            raise RuntimeError(
                f"SSB.predict_one called before fit() (or qid {qid!r} missing)"
            )
        mu_q = atom_to_mu_q(atom, qid)
        val = mu_q[self._best_source[qid]]
        if val is not None:
            return Prediction(answer=val, would_skip=False)
        # Random fallback on null source.
        return Prediction(
            answer=self._rng.choice(QUESTIONS[qid]["answer_space"]),
            would_skip=False,
        )

    def state_dict(self) -> dict:
        return {"best_source": dict(self._best_source)}

    def load_state_dict(self, state: dict) -> None:
        self._best_source = dict(state.get("best_source", {}))


class SSBGlobal(Method):
    """Single-source baseline using one globally best source for all questions.

    This legacy ablation is used only by the DGP-perturbation appendix. The
    main paper's SSB row uses :class:`SSB`, which chooses a best source per
    question.
    """

    name = "SSB-Global"
    requires_fit = True
    requires_calibration = False

    def __init__(self, seed: int | None = 42) -> None:
        self._best_source: str = SOURCE_NAMES[0]
        self._rng = random.Random(seed)

    def fit(self, records: Sequence[TrainingRecord]) -> None:
        best_score = -1.0
        best_source = SOURCE_NAMES[0]
        for src in SOURCE_NAMES:
            total_score = 0.0
            for qid in QUESTIONS:
                k = len(QUESTIONS[qid]["answer_space"])
                correct = 0
                n_null = 0
                for atom, gt in records:
                    if qid not in gt:
                        continue
                    mu_q = atom_to_mu_q(atom, qid)
                    val = mu_q[src]
                    if val is None:
                        n_null += 1
                    elif val == gt[qid]:
                        correct += 1
                total_score += correct + n_null / k
            if total_score > best_score:
                best_score = total_score
                best_source = src
        self._best_source = best_source

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        mu_q = atom_to_mu_q(atom, qid)
        val = mu_q[self._best_source]
        if val is not None:
            return Prediction(answer=val, would_skip=False)
        return Prediction(
            answer=self._rng.choice(QUESTIONS[qid]["answer_space"]),
            would_skip=False,
        )

    def state_dict(self) -> dict:
        return {"best_source": self._best_source}

    def load_state_dict(self, state: dict) -> None:
        self._best_source = str(state.get("best_source", SOURCE_NAMES[0]))


class SSBSelective(Method):
    """SSB with calibrated SKIP (C1 = null source, C2 = low agreement).

    ``seed`` defaults to 42 (paper canonical).
    """

    name = "SSBSelective"
    requires_fit = True
    requires_calibration = True

    def __init__(self, seed: int | None = 42) -> None:
        self._best_source: dict[str, str] = {}
        self._theta_agree: float = 0.0
        # No RNG is needed at predict time (C1 returns SKIP rather than
        # falling back to random), but we keep one for future symmetry.
        self._rng = random.Random(seed)

    def fit(self, records: Sequence[TrainingRecord]) -> None:
        self._best_source = _fit_best_source_per_question(records)

    def calibrate(self, records: Sequence[TrainingRecord]) -> None:
        """Grid-search `theta_agree` over [0, 1] step 0.05 to maximise F_0.5."""
        observations: list[tuple[float, float]] = []  # (agree_frac, is_correct)

        for atom, gt in records:
            for qid in QUESTIONS:
                if qid not in gt or qid not in self._best_source:
                    continue
                mu_q = atom_to_mu_q(atom, qid)
                src = self._best_source[qid]
                ans = mu_q[src]
                if ans is None:
                    continue  # C1 is unconditional, not threshold-tuned.
                active = [v for v in mu_q.values() if v is not None]
                if len(active) < 2:
                    continue  # Single active source always answers.
                agree = sum(1 for v in active if v == ans)
                agree_frac = agree / len(active)
                observations.append((agree_frac, float(ans == gt[qid])))

        best_f = -1.0
        best_theta = 0.0
        beta_sq = 0.5 ** 2
        for step in range(21):  # 0.00, 0.05, ..., 1.00
            theta = step * 0.05
            tp = fp = fn = 0.0
            for agree_frac, correct in observations:
                if agree_frac < theta:
                    fn += correct
                else:
                    tp += correct
                    fp += 1.0 - correct
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = beta_sq * prec + rec
            f_beta = ((1 + beta_sq) * prec * rec / denom) if denom > 0 else 0.0
            if f_beta > best_f:
                best_f = f_beta
                best_theta = theta
        self._theta_agree = best_theta

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        if qid not in self._best_source:
            raise RuntimeError(
                f"SSBSelective.predict_one called before fit() (or qid {qid!r} missing)"
            )
        mu_q = atom_to_mu_q(atom, qid)
        ans = mu_q[self._best_source[qid]]
        if ans is None:
            return Prediction(answer=SKIP_SENTINEL, would_skip=True)
        active = [v for v in mu_q.values() if v is not None]
        if len(active) >= 2:
            agree_frac = sum(1 for v in active if v == ans) / len(active)
            if agree_frac < self._theta_agree:
                return Prediction(answer=SKIP_SENTINEL, would_skip=True)
        return Prediction(answer=ans, would_skip=False)

    def state_dict(self) -> dict:
        return {
            "best_source": dict(self._best_source),
            "theta_agree": self._theta_agree,
        }

    def load_state_dict(self, state: dict) -> None:
        self._best_source = dict(state.get("best_source", {}))
        self._theta_agree = float(state.get("theta_agree", 0.0))
