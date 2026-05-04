"""ABF: Abductive Bias-aware Fusion.

Soft-mixture alignment over candidate answers, with MLE-trained
deflation weights, kernel sharpness, and bias-confidence mixture, plus
optional tiered SKIP calibrated by F_β grid search.

Score for one question Q with active sources S = {s : μ(s,Q) is not None}:

    ExplScore(v) = Σ_{s ∈ S} w_s · [π · κ_α(d_bias) + (1 - π) · κ_α(d_id)]

    w_s        = 1 - δ_s                          (objective_log fixed at δ=0)
    κ_α(d)     = max(0, 1 - α · d)                (truncated linear kernel)
    d_bias     = ord_dist(μ(s,Q), bias_predict(s, Q, v))
    d_id       = ord_dist(μ(s,Q), v)
    ord_dist   = |enc[a] - enc[b]| / (K-1)        (ordinal; Hamming = 1 for nominal)

Six learned parameters (train, MLE / L-BFGS-B):
    δ_prof, δ_plan, δ_self, δ_dev ∈ [0, 0.5];  α ∈ [0.5, 10];  π ∈ [0, 1].

Two calibrated parameters (cal split, F_β=0.5 grid; ABFSelective only):
    θ_E ∈ {0.0, 0.5, ..., 5.0}    θ_Δ ∈ {0.0, 0.2, ..., 2.0}.

SKIP rule (tiered by source count, ABFSelective only):
    |S| = 0 → SKIP                 (zero evidence)
    |S| = 1 → answer               (degenerate, never SKIP)
    |S| ≥ 2 → SKIP iff best < θ_E or (best - second) < θ_Δ.

Numerical equivalence with the prior reference implementation is verified
in ``tests/methods/test_abf.py`` on a real multi-difficulty persona slice.
The legacy ``_theta_ED`` serialization field (replaced by the structural
|S|-based rule) is intentionally dropped from the new state schema.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS, SOURCE_NAMES

from ._atom_adapter import atom_to_mu_q
from ._bias_model import bias_predict
from .base import SKIP_SENTINEL, Method, Prediction, TrainingRecord

# Sources whose deflation δ is learned; objective_log is the fixed anchor (δ=0).
_LEARNABLE_SOURCES: tuple[str, ...] = (
    "profile_ltm",
    "planner",
    "daily_self_report",
    "device_log",
)


# ── Adapter helpers (parallel to bcf.py / dsnbf.py; inlined per the byte-equivalence spec) ──


def _atom_to_mu_all(atom: ExtractedAtom) -> dict[str, dict[str, str | None]]:
    """Materialise per-question mu_q for every qid."""
    return {qid: atom_to_mu_q(atom, qid) for qid in QUESTIONS}


def _records_to_instances(records: Sequence[TrainingRecord]) -> list[dict]:
    """Project (ExtractedAtom, gt) records into legacy-shaped fit dicts.

    Inlined (rather than shared) to protect byte-equivalence across
    methods; see ``bcf.py`` / ``dsnbf.py`` for the parallel definitions.
    """
    out: list[dict] = []
    for atom, gt in records:
        out.append({
            "persona": atom.persona,
            "gt": dict(gt),
            "mu": _atom_to_mu_all(atom),
        })
    return out


# ── Distance ──────────────────────────────────────────────────────────────


def _ordinal_distance(a: str, b: str, qid: str) -> float:
    """Normalised distance between two answer values for question *qid*.

    Ordinal: |enc[a] - enc[b]| / (K - 1).
    Nominal answer-space, missing encoding, or a label outside the
    encoding (e.g. an edge option): Hamming, i.e. 0.0 if equal else 1.0.
    """
    if a == b:
        return 0.0
    q = QUESTIONS[qid]
    enc = q.get("ordinal_encoding")
    if enc is None or a not in enc or b not in enc:
        return 1.0
    K = len(enc)
    if K <= 1:
        return 0.0
    return abs(enc[a] - enc[b]) / (K - 1)


# ── ABF (no SKIP) ─────────────────────────────────────────────────────────


class ABF(Method):
    """Abductive Bias-aware Fusion: 6 MLE-learned params, no SKIP.

    Coverage = 100%. Reported as ``"ABF"`` in the paper.
    """

    name = "ABF"
    requires_fit = True
    requires_calibration = False

    def __init__(self, seed: int = 42) -> None:
        self._seed = int(seed)
        self._rng = random.Random(self._seed)
        # MLE-learned (populated by fit / load_state_dict).
        self._deltas: dict[str, float] = {s: 0.0 for s in SOURCE_NAMES}
        self._alpha: float = 2.0
        self._pi: float = 0.5

    # ---- scoring ----

    def _expl_score(
        self, qid: str, candidate: str, active: dict[str, str],
    ) -> float:
        """Compute ExplScore(v) for one candidate using current params."""
        alpha = self._alpha
        pi = self._pi
        total = 0.0
        for s, obs in active.items():
            w = 1.0 - self._deltas.get(s, 0.0)
            predicted = bias_predict(s, qid, candidate)
            d_b = _ordinal_distance(obs, predicted, qid)
            d_i = _ordinal_distance(obs, candidate, qid)
            k_b = max(0.0, 1.0 - alpha * d_b)
            k_i = max(0.0, 1.0 - alpha * d_i)
            total += w * (pi * k_b + (1.0 - pi) * k_i)
        return total

    # ---- fit (Phase 1: MLE for δ, α, π) ----

    def fit(self, records: Sequence[TrainingRecord]) -> None:
        """Learn (4δ + α + π) on train split via MLE (L-BFGS-B).

        Excludes per-question records with no active source or whose
        ground-truth label falls outside the answer space.
        """
        instances = _records_to_instances(records)
        source_order = list(SOURCE_NAMES)
        n_src = len(source_order)
        cand_lists = {qid: QUESTIONS[qid]["answer_space"] for qid in QUESTIONS}
        max_cands = max(len(cl) for cl in cand_lists.values())

        # ── Pre-compute fixed distance arrays (legacy byte-equiv pattern) ──
        rec_list: list[tuple[int, int]] = []
        d_bias_blocks: list[np.ndarray] = []
        d_id_blocks: list[np.ndarray] = []
        mask_blocks: list[np.ndarray] = []

        for inst in instances:
            for qid in QUESTIONS:
                mu_q = inst["mu"][qid]
                gt = inst["gt"][qid]
                active = {s: v for s, v in mu_q.items() if v is not None}
                if len(active) < 1:
                    continue

                candidates = cand_lists[qid]
                n_c = len(candidates)
                gt_idx = candidates.index(gt) if gt in candidates else -1
                if gt_idx < 0:
                    continue

                db = np.zeros((n_src, n_c))
                di = np.zeros((n_src, n_c))
                sm = np.zeros(n_src)

                for s, obs in active.items():
                    si = source_order.index(s)
                    sm[si] = 1.0
                    for ci, c in enumerate(candidates):
                        predicted = bias_predict(s, qid, c)
                        db[si, ci] = _ordinal_distance(obs, predicted, qid)
                        di[si, ci] = _ordinal_distance(obs, c, qid)

                rec_list.append((n_c, gt_idx))
                d_bias_blocks.append(db)
                d_id_blocks.append(di)
                mask_blocks.append(sm)

        n_rec = len(rec_list)
        if n_rec == 0:
            return

        d_bias_arr = np.zeros((n_rec, n_src, max_cands))
        d_id_arr = np.zeros((n_rec, n_src, max_cands))
        source_mask = np.zeros((n_rec, n_src))
        valid_mask = np.zeros((n_rec, max_cands), dtype=bool)
        gt_arr = np.zeros(n_rec, dtype=np.int32)

        for r, ((n_c, gi), db, di, sm) in enumerate(
            zip(rec_list, d_bias_blocks, d_id_blocks, mask_blocks)
        ):
            d_bias_arr[r, :, :n_c] = db
            d_id_arr[r, :, :n_c] = di
            source_mask[r] = sm
            valid_mask[r, :n_c] = True
            gt_arr[r] = gi

        learnable_si = [source_order.index(s) for s in _LEARNABLE_SOURCES]

        def neg_log_lik(params: np.ndarray) -> float:
            delta_vals = params[:4]
            alpha = params[4]
            pi = params[5]

            w = np.ones(n_src)
            for i, si in enumerate(learnable_si):
                w[si] = 1.0 - delta_vals[i]
            # objective_log keeps weight 1.0.

            kb = np.clip(1.0 - alpha * d_bias_arr, 0.0, None)
            ki = np.clip(1.0 - alpha * d_id_arr, 0.0, None)
            kernel = pi * kb + (1.0 - pi) * ki
            weighted = kernel * (w[None, :, None] * source_mask[:, :, None])
            expl = weighted.sum(axis=1)

            expl = np.where(valid_mask, expl, -1e30)
            log_norm = logsumexp(expl, axis=1)
            log_p_gt = expl[np.arange(n_rec), gt_arr] - log_norm
            return -log_p_gt.mean()

        x0 = np.array([0.2, 0.2, 0.2, 0.2, 2.0, 0.5])
        bounds = (
            [(0.0, 0.5)] * 4
            + [(0.5, 10.0)]
            + [(0.0, 1.0)]
        )

        result = minimize(
            neg_log_lik, x0, method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 200},
        )

        best = result.x
        self._deltas = {s: float(best[i]) for i, s in enumerate(_LEARNABLE_SOURCES)}
        self._deltas["objective_log"] = 0.0
        self._alpha = float(best[4])
        self._pi = float(best[5])

    # ---- predict ----

    def _predict_label(self, qid: str, mu_q: dict[str, str | None]) -> str:
        candidates = QUESTIONS[qid]["answer_space"]
        active = {s: v for s, v in mu_q.items() if v is not None}

        if not active:
            return self._rng.choice(candidates)

        scores = [self._expl_score(qid, c, active) for c in candidates]
        best_score = max(scores)
        winners = [i for i, sc in enumerate(scores) if sc == best_score]
        if len(winners) == 1:
            return candidates[winners[0]]
        return self._rng.choice(sorted([candidates[i] for i in winners]))

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        mu_q = atom_to_mu_q(atom, qid)
        return Prediction(
            answer=self._predict_label(qid, mu_q), would_skip=False,
        )

    # ---- state persistence ----

    def state_dict(self) -> dict:
        return {
            "seed": self._seed,
            "deltas": dict(self._deltas),
            "alpha": self._alpha,
            "pi": self._pi,
        }

    def load_state_dict(self, state: dict) -> None:
        self._seed = int(state.get("seed", self._seed))
        self._rng = random.Random(self._seed)
        deltas = state.get("deltas", {})
        self._deltas = {s: float(deltas.get(s, 0.0)) for s in SOURCE_NAMES}
        self._alpha = float(state.get("alpha", 2.0))
        self._pi = float(state.get("pi", 0.5))


# ── ABFSelective: tiered SKIP with calibrated θ_E, θ_Δ ───────────────────


class ABFSelective(ABF):
    """ABF with tiered SKIP (8 params total: 6 MLE + 2 cal).

    SKIP rule:
        |S| = 0 → SKIP (no evidence)
        |S| = 1 → answer (degenerate)
        |S| ≥ 2 → SKIP iff best < θ_E or (best - second) < θ_Δ.
    """

    name = "ABF+SKIP"
    requires_fit = True
    requires_calibration = True

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self._theta_E: float = 0.0
        self._theta_delta: float = 0.0

    # ---- calibrate (Phase 2: F_β=0.5 grid over θ_E × θ_Δ) ----

    def calibrate(self, records: Sequence[TrainingRecord]) -> None:
        """Learn θ_E, θ_Δ on the cal split (multi-source records only).

        F_β=0.5 (precision-oriented) on per-(persona, qid) records with
        |S|≥2. Single/zero-source records are excluded because the
        structural rule already determines their action.
        """
        cal_instances = _records_to_instances(records)

        records_obs: list[tuple[float, float, float]] = []  # (best, margin, frac_correct)
        for inst in cal_instances:
            for qid in QUESTIONS:
                mu_q = inst["mu"][qid]
                gt = inst["gt"][qid]
                active = {s: v for s, v in mu_q.items() if v is not None}
                if len(active) < 2:
                    continue

                candidates = QUESTIONS[qid]["answer_space"]
                scores = [self._expl_score(qid, c, active) for c in candidates]

                sorted_desc = sorted(scores, reverse=True)
                best = sorted_desc[0]
                second = sorted_desc[1] if len(sorted_desc) > 1 else 0.0
                margin = best - second

                winners = [i for i, sc in enumerate(scores) if sc == best]
                gt_in = any(candidates[i] == gt for i in winners)
                frac_correct = (1.0 / len(winners)) if gt_in else 0.0

                records_obs.append((best, margin, frac_correct))

        theta_E_vals = [i * 0.5 for i in range(11)]   # [0.0, 5.0]
        theta_D_vals = [i * 0.2 for i in range(11)]   # [0.0, 2.0]

        best_score = -1.0
        best_theta = (0.0, 0.0)
        beta = 0.5
        beta2 = beta * beta

        for te in theta_E_vals:
            for td in theta_D_vals:
                tp = fp = fn = 0.0
                for expl_r, margin_r, frac_c in records_obs:
                    if (expl_r < te) or (margin_r < td):
                        fn += frac_c
                    else:
                        tp += frac_c
                        fp += 1.0 - frac_c

                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                denom = beta2 * prec + rec
                f_beta = ((1 + beta2) * prec * rec / denom) if denom > 0 else 0.0
                if f_beta > best_score:
                    best_score = f_beta
                    best_theta = (te, td)

        self._theta_E, self._theta_delta = best_theta

    # ---- predict (overrides parent to add tiered SKIP) ----

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        mu_q = atom_to_mu_q(atom, qid)
        candidates = QUESTIONS[qid]["answer_space"]
        active = {s: v for s, v in mu_q.items() if v is not None}
        n_active = len(active)

        # Tier 1: zero evidence → always SKIP.
        if n_active == 0:
            return Prediction(answer=SKIP_SENTINEL, would_skip=True)

        scores = [self._expl_score(qid, c, active) for c in candidates]
        best_score = max(scores)

        # Tier 3: multi-source → calibrated thresholds. (Tier 2 |S|=1 falls
        # through directly to the answer path.)
        if n_active >= 2:
            if best_score < self._theta_E:
                return Prediction(answer=SKIP_SENTINEL, would_skip=True)
            sorted_desc = sorted(scores, reverse=True)
            second = sorted_desc[1] if len(sorted_desc) > 1 else 0.0
            if best_score - second < self._theta_delta:
                return Prediction(answer=SKIP_SENTINEL, would_skip=True)

        winners = [i for i, sc in enumerate(scores) if sc == best_score]
        if len(winners) == 1:
            label = candidates[winners[0]]
        else:
            label = self._rng.choice(sorted([candidates[i] for i in winners]))
        return Prediction(answer=label, would_skip=False)

    # ---- state persistence (extends parent) ----

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["theta_E"] = self._theta_E
        state["theta_delta"] = self._theta_delta
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self._theta_E = float(state.get("theta_E", 0.0))
        self._theta_delta = float(state.get("theta_delta", 0.0))
