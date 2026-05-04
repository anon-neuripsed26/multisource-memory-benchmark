"""NBF: Naive Bayes Fusion (paper Table 4 + Table 5 with selective abstention).

Per-question per-source confusion matrices ``P(source_says_v | GT = w)`` are
estimated from the train split with Laplace smoothing (α=1). At test time
we compute the (Bayes-rule) posterior over GT for one question, then either:

* ``NBF``           — return the MAP answer (no abstention).
* ``NBFSelective``  — abstain when at least 2 sources are active and the
                      posterior margin (top-1 minus top-2 normalised
                      probability) falls below a calibrated ``θ_margin``.

Both variants calibrate the emission temperature ``T`` on the cal split.
``NBFSelective`` jointly grids ``(T, θ_margin)`` to maximise F_{0.5}.

Byte-equivalent to the legacy v1.0 reference
``NaiveBayesFusion`` (skip ∈ {False, True}) when run with the default
``alpha=1.0``, ``weight_exp=0.0``, ``acc_threshold=0.0`` knobs.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS, SOURCE_NAMES

from ._atom_adapter import atom_to_mu_q
from .base import SKIP_SENTINEL, Method, Prediction, TrainingRecord


def _atom_to_mu_all(atom: ExtractedAtom) -> dict[str, dict[str, str | None]]:
    return {qid: atom_to_mu_q(atom, qid) for qid in QUESTIONS}


def _records_to_instances(records: Sequence[TrainingRecord]) -> list[dict]:
    """Inlined (per task spec) — do not extract to a shared helper."""
    out: list[dict] = []
    for atom, gt in records:
        out.append({
            "persona": atom.persona,
            "gt": dict(gt),
            "mu": _atom_to_mu_all(atom),
        })
    return out


class _NBFCore:
    """Shared state + fit/predict logic for NBF and NBFSelective."""

    def __init__(
        self,
        seed: int = 42,
        alpha: float = 1.0,
        weight_exp: float = 0.0,
        acc_threshold: float = 0.0,
    ) -> None:
        self._seed = int(seed)
        self._alpha = float(alpha)
        self._weight_exp = float(weight_exp)
        self._acc_threshold = float(acc_threshold)
        self._rng = random.Random(self._seed)

        self._log_emit: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        self._log_prior: dict[str, dict[str, float]] = {}
        self._source_acc: dict[str, dict[str, float]] = {}
        self._theta_margin: float = 0.0
        self._temperature: float = 1.0

    # ---- fit ----

    def _fit_instances(self, train_instances: list[dict]) -> None:
        for qid in QUESTIONS:
            answer_space = QUESTIONS[qid]["answer_space"]
            K = len(answer_space)

            # Prior over GT.
            gt_marginal: dict[str, int] = {w: 0 for w in answer_space}
            for inst in train_instances:
                gt_marginal[inst["gt"][qid]] += 1
            total_gt = sum(gt_marginal.values())
            self._log_prior[qid] = {
                w: math.log((gt_marginal[w] + 1) / (total_gt + K))
                for w in answer_space
            }

            # Per-source accuracy (used by weight_exp / acc_threshold).
            self._source_acc[qid] = {}
            for src in SOURCE_NAMES:
                correct = 0
                n_nonnull = 0
                for inst in train_instances:
                    mu_val = inst["mu"][qid].get(src)
                    if mu_val is not None:
                        n_nonnull += 1
                        if mu_val == inst["gt"][qid]:
                            correct += 1
                self._source_acc[qid][src] = (
                    correct / n_nonnull if n_nonnull > 0 else 1.0 / K
                )

            # Per-source confusion matrices, Laplace-smoothed.
            self._log_emit[qid] = {}
            for src in SOURCE_NAMES:
                counts: dict[str, dict[str, float]] = {
                    v: {w: self._alpha for w in answer_space}
                    for v in answer_space
                }
                gt_denom: dict[str, float] = {
                    w: self._alpha * K for w in answer_space
                }
                for inst in train_instances:
                    mu_val = inst["mu"][qid].get(src)
                    gt_val = inst["gt"][qid]
                    if mu_val is not None:
                        counts[mu_val][gt_val] += 1.0
                        gt_denom[gt_val] += 1.0

                self._log_emit[qid][src] = {}
                for v in answer_space:
                    self._log_emit[qid][src][v] = {}
                    for w in answer_space:
                        p = counts[v][w] / gt_denom[w]
                        self._log_emit[qid][src][v][w] = math.log(p)

    # ---- inference primitives ----

    def _log_posterior(
        self, qid: str, mu_q: dict[str, str | None],
    ) -> tuple[dict[str, float], int]:
        answer_space = QUESTIONS[qid]["answer_space"]
        log_posts: dict[str, float] = {}
        for w in answer_space:
            lp = self._log_prior[qid][w]
            for src in SOURCE_NAMES:
                obs = mu_q.get(src)
                if obs is None:
                    continue
                src_acc = self._source_acc[qid].get(src, 0.33)
                if src_acc < self._acc_threshold:
                    continue
                weight = src_acc ** self._weight_exp if self._weight_exp > 0 else 1.0
                lp += weight * self._log_emit[qid][src][obs][w] * self._temperature
            log_posts[w] = lp

        n_active = sum(
            1 for s in SOURCE_NAMES
            if mu_q.get(s) is not None
            and self._source_acc[qid].get(s, 0.33) >= self._acc_threshold
        )
        return log_posts, n_active

    @staticmethod
    def _posterior_margin(log_posts: dict[str, float]) -> float:
        vals = sorted(log_posts.values(), reverse=True)
        if len(vals) < 2:
            return 1.0
        max_lp = vals[0]
        probs = [math.exp(v - max_lp) for v in vals]
        total = sum(probs)
        probs = [p / total for p in probs]
        return probs[0] - probs[1]

    # ---- state persistence ----

    def _state_dict(self) -> dict:
        return {
            "seed": self._seed,
            "alpha": self._alpha,
            "weight_exp": self._weight_exp,
            "acc_threshold": self._acc_threshold,
            "temperature": self._temperature,
            "theta_margin": self._theta_margin,
            "log_emit": self._log_emit,
            "log_prior": self._log_prior,
            "source_acc": self._source_acc,
        }

    def _load_state_dict(self, state: dict) -> None:
        self._seed = int(state.get("seed", self._seed))
        self._alpha = float(state.get("alpha", 1.0))
        self._weight_exp = float(state.get("weight_exp", 0.0))
        self._acc_threshold = float(state.get("acc_threshold", 0.0))
        self._temperature = float(state.get("temperature", 1.0))
        self._theta_margin = float(state.get("theta_margin", 0.0))
        self._log_emit = state.get("log_emit", {})
        self._log_prior = state.get("log_prior", {})
        self._source_acc = state.get("source_acc", {})
        self._rng = random.Random(self._seed)


class NBF(_NBFCore, Method):
    """Naive Bayes Fusion, no abstention. Calibrates emission temperature only."""

    name = "NBF"
    requires_fit = True
    requires_calibration = True

    def fit(self, records: Sequence[TrainingRecord]) -> None:
        self._fit_instances(_records_to_instances(records))

    def calibrate(self, records: Sequence[TrainingRecord]) -> None:
        cal_instances = _records_to_instances(records)
        best_acc = -1.0
        best_T = 1.0
        for T_10 in range(1, 31):  # T ∈ {0.1, 0.2, ..., 3.0}
            T = T_10 * 0.1
            self._temperature = T
            q_correct: dict[str, int] = {}
            q_total: dict[str, int] = {}
            for inst in cal_instances:
                for qid in QUESTIONS:
                    q_total[qid] = q_total.get(qid, 0) + 1
                    log_posts, _ = self._log_posterior(qid, inst["mu"][qid])
                    best_w = max(log_posts, key=log_posts.get)
                    if best_w == inst["gt"][qid]:
                        q_correct[qid] = q_correct.get(qid, 0) + 1
            macro = sum(
                q_correct.get(q, 0) / q_total[q] for q in q_total
            ) / len(q_total)
            if macro > best_acc:
                best_acc = macro
                best_T = T
        self._temperature = best_T

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        if not self._log_emit:
            raise RuntimeError("NBF.predict_one called before fit()")
        mu_q = atom_to_mu_q(atom, qid)
        log_posts, _ = self._log_posterior(qid, mu_q)
        best_w = max(log_posts, key=log_posts.get)
        return Prediction(answer=best_w, would_skip=False)

    def state_dict(self) -> dict:
        return self._state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._load_state_dict(state)


class NBFSelective(_NBFCore, Method):
    """Naive Bayes Fusion with calibrated SKIP (paper Table 5).

    SKIP is gated on ``n_active >= 2``: single-source cases always answer.
    """

    name = "NBF+SKIP"
    requires_fit = True
    requires_calibration = True

    def fit(self, records: Sequence[TrainingRecord]) -> None:
        self._fit_instances(_records_to_instances(records))

    def calibrate(self, records: Sequence[TrainingRecord]) -> None:
        cal_instances = _records_to_instances(records)
        best_f05 = -1.0
        best_params = (1.0, 0.0)

        for T_10 in range(1, 31):
            T = T_10 * 0.1
            self._temperature = T

            records_obs: list[tuple[float, bool, int]] = []
            for inst in cal_instances:
                for qid in QUESTIONS:
                    log_posts, n_active = self._log_posterior(
                        qid, inst["mu"][qid])
                    best_w = max(log_posts, key=log_posts.get)
                    margin = self._posterior_margin(log_posts)
                    correct = (best_w == inst["gt"][qid])
                    records_obs.append((margin, correct, n_active))

            for theta100 in range(101):
                theta = theta100 * 0.01
                tp = fp = fn = 0
                for margin, correct, n_act in records_obs:
                    if n_act >= 2 and margin < theta:
                        fn += 1
                    elif correct:
                        tp += 1
                    else:
                        fp += 1
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                denom = 0.25 * prec + rec
                f05 = 1.25 * prec * rec / denom if denom > 0 else 0.0
                if f05 > best_f05:
                    best_f05 = f05
                    best_params = (T, theta)

        self._temperature, self._theta_margin = best_params

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        if not self._log_emit:
            raise RuntimeError("NBFSelective.predict_one called before fit()")
        mu_q = atom_to_mu_q(atom, qid)
        log_posts, n_active = self._log_posterior(qid, mu_q)
        best_w = max(log_posts, key=log_posts.get)

        if n_active >= 2:
            margin = self._posterior_margin(log_posts)
            if margin < self._theta_margin:
                return Prediction(answer=SKIP_SENTINEL, would_skip=True)
        return Prediction(answer=best_w, would_skip=False)

    def state_dict(self) -> dict:
        return self._state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._load_state_dict(state)
