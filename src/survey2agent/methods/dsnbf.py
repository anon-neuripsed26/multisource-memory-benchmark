"""DSNBF: Difficulty-Stratified Naive Bayes Fusion (T2 fusion, paper flagship).

Per-question confusion matrices saturate around 80% accuracy because they
ignore cross-question persona-level structure. DSNBF lifts this ceiling by
maintaining one *global* and three *difficulty-specific* (stable /
temporal_shift / stated_vs_revealed) confusion-matrix sets, then for each
test persona infers a soft posterior over the persona's difficulty class
from all 18 observations and blends the global and difficulty-weighted
posteriors via a calibrated weight `gw`.

Calibrated parameters (cal split, grid search on macro accuracy):
    T      - emission temperature (sharpens / softens source evidence)
    T_diff - difficulty-estimation temperature
    gw     - global-vs-stratified blend in [0, 1]   (1.0 = pure NBF)
    eta    - hierarchical Dirichlet strength (per-difficulty matrices
             shrink toward global emission as eta grows)
    theta  - SKIP margin (DSNBFSelective only): abstain when the gap
             between top-1 and top-2 combined probabilities < theta

Difficulty labels for train/calibration personas are supplied by the official
evaluation runner from persona-spec metadata. They are fit-time metadata for
the supervised resolver, not prediction inputs for the held-out target
persona. A persona-id fallback is retained only for legacy direct tuple inputs.

Numerical equivalence with v1.0 reference
(class ``DiffStratNBF``) is verified by a byte-equivalence script during
migration; both use Laplace ``alpha=1`` for the global matrix and a
hierarchical Dirichlet (``alpha_d=eta``) for per-difficulty matrices.
"""

from __future__ import annotations

import math
import random
import warnings
from collections import Counter
from typing import Sequence

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS, SOURCE_NAMES

from ._atom_adapter import atom_to_mu_q
from .base import SKIP_SENTINEL, Method, Prediction, TrainingRecord

_DIFFS: tuple[str, ...] = ("stable", "temporal_shift", "stated_vs_revealed")
_PREFIX_MAP: dict[str, str] = {
    "stable": "stable",
    "shift": "temporal_shift",
    "stated": "stated_vs_revealed",
}


def _persona_difficulty(persona_id: str) -> str:
    """Legacy fallback: derive difficulty from a ``bench_<prefix>`` persona id.

    The official evaluation runner supplies difficulty metadata explicitly.
    This fallback supports older direct ``(atom, gt)`` calls and assigns
    non-conforming ids to ``"stable"`` defensively so the global matrix still
    receives the persona's contribution.
    """
    parts = persona_id.split("_")
    if len(parts) >= 2 and parts[1] in _PREFIX_MAP:
        return _PREFIX_MAP[parts[1]]
    warnings.warn(
        (
            f"DSNBF: persona id {persona_id!r} does not follow the "
            "`bench_<stable|shift|stated>_NNN_<name>` convention; "
            "falling back to difficulty class 'stable'. Difficulty "
            "stratification is silently disabled for this persona. "
            "See DATASHEET.md § Persona Identifier Convention."
        ),
        RuntimeWarning,
        stacklevel=2,
    )
    return "stable"


def _record_difficulty(record: TrainingRecord, atom: ExtractedAtom) -> str:
    """Return fit/cal difficulty metadata, with a legacy persona-id fallback."""
    difficulty = getattr(record, "difficulty_class", None)
    if difficulty is None:
        return _persona_difficulty(atom.persona)
    if difficulty not in _DIFFS:
        raise ValueError(
            f"DSNBF: unknown difficulty_class {difficulty!r}; expected one of {_DIFFS}"
        )
    return str(difficulty)


def _atom_to_mu_all(atom: ExtractedAtom) -> dict[str, dict[str, str | None]]:
    """Materialise per-question mu_q for every qid (used by prepare_persona)."""
    return {qid: atom_to_mu_q(atom, qid) for qid in QUESTIONS}


def _records_to_instances(
    records: Sequence[TrainingRecord],
) -> list[dict]:
    """Project (ExtractedAtom, gt) records into legacy-shaped fit dicts."""
    out: list[dict] = []
    for record in records:
        atom, gt = record
        out.append({
            "persona": atom.persona,
            "gt": dict(gt),
            "mu": _atom_to_mu_all(atom),
            "difficulty": _record_difficulty(record, atom),
        })
    return out


# ── Matrix construction (parallels DiffStratNBF._build / _build_hier) ──────


def _build_global(
    instances: list[dict], alpha: float,
) -> tuple[dict, dict]:
    """Laplace-smoothed global emission tables and GT log-priors."""
    emit: dict = {}
    prior: dict = {}
    n = len(instances)
    for qid in QUESTIONS:
        ans = QUESTIONS[qid]["answer_space"]
        K = len(ans)
        gc: dict[str, int] = {w: 0 for w in ans}
        for inst in instances:
            gc[inst["gt"][qid]] += 1
        prior[qid] = {w: math.log((gc[w] + 1) / (n + K)) for w in ans}
        emit[qid] = {}
        for src in SOURCE_NAMES:
            cts: dict[str, dict[str, float]] = {
                v: {w: alpha for w in ans} for v in ans
            }
            gd: dict[str, float] = {w: alpha * K for w in ans}
            for inst in instances:
                obs = inst["mu"][qid].get(src)
                gt_val = inst["gt"][qid]
                if obs is not None:
                    cts[obs][gt_val] += 1.0
                    gd[gt_val] += 1.0
            emit[qid][src] = {}
            for v in ans:
                emit[qid][src][v] = {}
                for w in ans:
                    emit[qid][src][v][w] = math.log(cts[v][w] / gd[w])
    return emit, prior


def _build_hier(
    instances: list[dict], global_emit: dict, eta: float,
) -> tuple[dict, dict]:
    """Per-difficulty emission tables with global emission as Dirichlet prior."""
    emit: dict = {}
    prior: dict = {}
    n = len(instances)
    for qid in QUESTIONS:
        ans = QUESTIONS[qid]["answer_space"]
        K = len(ans)
        gc: dict[str, int] = {w: 0 for w in ans}
        for inst in instances:
            gc[inst["gt"][qid]] += 1
        prior[qid] = {w: math.log((gc[w] + 1) / (n + K)) for w in ans}
        emit[qid] = {}
        for src in SOURCE_NAMES:
            cts: dict[str, dict[str, float]] = {}
            for v in ans:
                cts[v] = {}
                for w in ans:
                    g_prob = math.exp(global_emit[qid][src][v][w])
                    cts[v][w] = eta * g_prob
            gd: dict[str, float] = {w: eta for w in ans}
            for inst in instances:
                obs = inst["mu"][qid].get(src)
                gt_val = inst["gt"][qid]
                if obs is not None:
                    cts[obs][gt_val] += 1.0
                    gd[gt_val] += 1.0
            emit[qid][src] = {}
            for v in ans:
                emit[qid][src][v] = {}
                for w in ans:
                    emit[qid][src][v][w] = math.log(cts[v][w] / gd[w])
    return emit, prior


# ── DSNBF core ────────────────────────────────────────────────────────────


class DSNBF(Method):
    """Difficulty-Stratified Naive Bayes Fusion (no SKIP)."""

    name = "DSNBF"
    requires_fit = True
    requires_calibration = True

    def __init__(
        self,
        seed: int = 42,
        alpha: float = 1.0,
        alpha_d: float = 5.0,
    ) -> None:
        # Smoothing constants (legacy defaults: alpha=1.0 Laplace, alpha_d=5.0 Dirichlet).
        self._alpha: float = alpha
        self._alpha_d: float = alpha_d

        # Calibrated hyperparameters (defaults match legacy pre-calibration state).
        self._T: float = 1.0          # source-emission temperature
        self._T_diff: float = 1.0     # difficulty-inference temperature
        self._gw: float = 0.5         # global vs. difficulty-stratified blend
        self._theta: float = 0.0      # SKIP margin (used by Selective subclass)

        # Learned tables (populated by fit / load_state_dict).
        self._emit_g: dict = {}       # global log-emission: qid → src → v → w → log p
        self._prior_g: dict = {}      # global log-prior:    qid → w → log p
        self._emit_d: dict = {}       # per-difficulty emission: diff → qid → src → v → w
        self._prior_d: dict = {}      # per-difficulty prior:    diff → qid → w
        self._diff_log_prior: dict = {}  # diff → log P(diff) on train

        # Train cache enables eta-grid recalibration (DSNBFSelective only).
        self._train_cache: list[dict] | None = None

        # Per-persona inference cache (avoids redoing prepare_persona per qid).
        self._cached_persona: str | None = None
        self._diff_probs: dict[str, float] | None = None

        self._rng = random.Random(seed)

    # ---- fit ----

    def fit(self, records: Sequence[TrainingRecord]) -> None:
        instances = _records_to_instances(records)
        self._train_cache = instances
        self._emit_g, self._prior_g = _build_global(instances, self._alpha)

        dc = Counter(i["difficulty"] for i in instances)
        n = len(instances)
        self._diff_log_prior = {
            d: math.log((dc.get(d, 0) + 1) / (n + len(_DIFFS))) for d in _DIFFS
        }
        self._emit_d = {}
        self._prior_d = {}
        for d in _DIFFS:
            di = [i for i in instances if i["difficulty"] == d]
            self._emit_d[d], self._prior_d[d] = _build_hier(
                di, self._emit_g, self._alpha_d,
            )
        # Invalidate any prior persona cache.
        self._cached_persona = None
        self._diff_probs = None

    # ---- prepare_persona (P(difficulty | all observations)) ----

    def _prepare_persona(
        self, mu_all: dict[str, dict[str, str | None]],
    ) -> None:
        lps: dict[str, float] = {}
        for d in _DIFFS:
            lp = self._diff_log_prior[d]
            for qid in QUESTIONS:
                ans = QUESTIONS[qid]["answer_space"]
                terms: list[float] = []
                for w in ans:
                    t = self._prior_d[d][qid][w]
                    for src in SOURCE_NAMES:
                        obs = mu_all[qid].get(src)
                        if obs is not None:
                            t += self._emit_d[d][qid][src][obs][w] * self._T_diff
                    terms.append(t)
                mx = max(terms)
                lp += mx + math.log(sum(math.exp(t - mx) for t in terms))
            lps[d] = lp
        mx = max(lps.values())
        raw = {d: math.exp(v - mx) for d, v in lps.items()}
        s = sum(raw.values())
        self._diff_probs = {d: p / s for d, p in raw.items()}

    # ---- predict ----

    def _combined_probs(
        self, qid: str, mu_q: dict[str, str | None],
    ) -> dict[str, float]:
        """Blend global and difficulty-weighted posteriors for one question."""
        ans = QUESTIONS[qid]["answer_space"]
        dp = self._diff_probs or {d: 1.0 / len(_DIFFS) for d in _DIFFS}

        lp_g: dict[str, float] = {}
        for w in ans:
            v = self._prior_g[qid][w]
            for src in SOURCE_NAMES:
                obs = mu_q.get(src)
                if obs is not None:
                    v += self._emit_g[qid][src][obs][w] * self._T
            lp_g[w] = v
        mx_g = max(lp_g.values())
        pg = {w: math.exp(lp_g[w] - mx_g) for w in ans}
        sg = sum(pg.values())

        dw: dict[str, float] = {w: 0.0 for w in ans}
        for d in _DIFFS:
            lp_d: dict[str, float] = {}
            for w in ans:
                v = self._prior_d[d][qid][w]
                for src in SOURCE_NAMES:
                    obs = mu_q.get(src)
                    if obs is not None:
                        v += self._emit_d[d][qid][src][obs][w] * self._T
                lp_d[w] = v
            mx_d = max(lp_d.values())
            pd = {w: math.exp(lp_d[w] - mx_d) for w in ans}
            sd = sum(pd.values())
            for w in ans:
                dw[w] += dp[d] * pd[w] / sd

        return {
            w: self._gw * pg[w] / sg + (1.0 - self._gw) * dw[w] for w in ans
        }

    def _predict_label(self, qid: str, mu_q: dict[str, str | None]) -> str:
        if not self._emit_g:
            raise RuntimeError(
                f"{type(self).__name__}.predict_one called before fit()"
            )
        combined = self._combined_probs(qid, mu_q)
        return max(combined, key=combined.get)  # type: ignore[arg-type]

    def _ensure_persona_prepared(self, atom: ExtractedAtom) -> None:
        if self._cached_persona == atom.persona and self._diff_probs is not None:
            return
        self._prepare_persona(_atom_to_mu_all(atom))
        self._cached_persona = atom.persona

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        self._ensure_persona_prepared(atom)
        mu_q = atom_to_mu_q(atom, qid)
        ans = self._predict_label(qid, mu_q)
        return Prediction(answer=ans, would_skip=False)

    # ---- state persistence ----

    def state_dict(self) -> dict:
        return {
            "alpha": self._alpha,
            "alpha_d": self._alpha_d,
            "T": self._T,
            "T_diff": self._T_diff,
            "gw": self._gw,
            "theta": self._theta,
            "emit_g": self._emit_g,
            "prior_g": self._prior_g,
            "emit_d": self._emit_d,
            "prior_d": self._prior_d,
            "diff_log_prior": self._diff_log_prior,
        }

    def load_state_dict(self, state: dict) -> None:
        self._alpha = float(state.get("alpha", 1.0))
        self._alpha_d = float(state.get("alpha_d", 5.0))
        self._T = float(state.get("T", 1.0))
        self._T_diff = float(state.get("T_diff", 1.0))
        self._gw = float(state.get("gw", 0.5))
        self._theta = float(state.get("theta", 0.0))
        self._emit_g = state.get("emit_g", {})
        self._prior_g = state.get("prior_g", {})
        self._emit_d = state.get("emit_d", {})
        self._prior_d = state.get("prior_d", {})
        self._diff_log_prior = state.get("diff_log_prior", {})
        self._cached_persona = None
        self._diff_probs = None

    # ---- calibrate (eta × Td × T × gw grid; mirrors DiffStratNBF) ----

    def _precompute_cal(self, cal_instances: list[dict]) -> list[dict]:
        precomp: list[dict] = []
        for inst in cal_instances:
            pc: dict = {"gt": inst["gt"]}
            eg: dict = {}
            for qid in QUESTIONS:
                eg[qid] = {}
                for w in QUESTIONS[qid]["answer_space"]:
                    s = 0.0
                    for src in SOURCE_NAMES:
                        obs = inst["mu"][qid].get(src)
                        if obs is not None:
                            s += self._emit_g[qid][src][obs][w]
                    eg[qid][w] = s
            pc["eg"] = eg
            ed: dict = {}
            for qid in QUESTIONS:
                ed[qid] = {}
                for d in _DIFFS:
                    ed[qid][d] = {}
                    for w in QUESTIONS[qid]["answer_space"]:
                        s = 0.0
                        for src in SOURCE_NAMES:
                            obs = inst["mu"][qid].get(src)
                            if obs is not None:
                                s += self._emit_d[d][qid][src][obs][w]
                        ed[qid][d][w] = s
            pc["ed"] = ed
            pc["n_act"] = {
                qid: sum(
                    1 for s in SOURCE_NAMES
                    if inst["mu"][qid].get(s) is not None
                )
                for qid in QUESTIONS
            }
            precomp.append(pc)
        return precomp

    def _grid_search(
        self, precomp: list[dict],
    ) -> tuple[float, tuple[float, float, float]]:
        T_grid = [t * 0.1 for t in range(3, 21)]
        Td_grid = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        gw_grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

        best_macro = -1.0
        best_p: tuple[float, float, float] = (1.0, 1.0, 0.5)

        for Td in Td_grid:
            all_dp: list[dict[str, float]] = []
            for pc in precomp:
                dlps: dict[str, float] = {}
                for d in _DIFFS:
                    lp = self._diff_log_prior[d]
                    for qid in QUESTIONS:
                        ans = QUESTIONS[qid]["answer_space"]
                        terms = [
                            self._prior_d[d][qid][w] + Td * pc["ed"][qid][d][w]
                            for w in ans
                        ]
                        mx = max(terms)
                        lp += mx + math.log(
                            sum(math.exp(t - mx) for t in terms)
                        )
                    dlps[d] = lp
                mx = max(dlps.values())
                raw = {d: math.exp(v - mx) for d, v in dlps.items()}
                st = sum(raw.values())
                all_dp.append({d: p / st for d, p in raw.items()})

            for T in T_grid:
                all_pg: list[dict] = []
                all_pd: list[dict] = []
                for idx, pc in enumerate(precomp):
                    pg_i: dict = {}
                    pd_i: dict = {}
                    ddp = all_dp[idx]
                    for qid in QUESTIONS:
                        ans = QUESTIONS[qid]["answer_space"]
                        lpg = {
                            w: self._prior_g[qid][w] + T * pc["eg"][qid][w]
                            for w in ans
                        }
                        mxg = max(lpg.values())
                        prg = {w: math.exp(lpg[w] - mxg) for w in ans}
                        stg = sum(prg.values())
                        pg_i[qid] = {w: prg[w] / stg for w in ans}
                        dwq: dict[str, float] = {w: 0.0 for w in ans}
                        for d in _DIFFS:
                            lpd = {
                                w: self._prior_d[d][qid][w]
                                + T * pc["ed"][qid][d][w]
                                for w in ans
                            }
                            mxd = max(lpd.values())
                            prd = {w: math.exp(lpd[w] - mxd) for w in ans}
                            std_ = sum(prd.values())
                            for w in ans:
                                dwq[w] += ddp[d] * prd[w] / std_
                        pd_i[qid] = dwq
                    all_pg.append(pg_i)
                    all_pd.append(pd_i)

                for gw in gw_grid:
                    qc: dict[str, int] = {}
                    qt: dict[str, int] = {}
                    for idx, pc in enumerate(precomp):
                        for qid in QUESTIONS:
                            ans = QUESTIONS[qid]["answer_space"]
                            qt[qid] = qt.get(qid, 0) + 1
                            cb = {
                                w: gw * all_pg[idx][qid][w]
                                + (1.0 - gw) * all_pd[idx][qid][w]
                                for w in ans
                            }
                            pred = max(cb, key=cb.get)  # type: ignore[arg-type]
                            if pred == pc["gt"][qid]:
                                qc[qid] = qc.get(qid, 0) + 1
                    macro = sum(qc.get(q, 0) / qt[q] for q in qt) / len(qt)
                    if macro > best_macro:
                        best_macro = macro
                        best_p = (T, Td, gw)

        return best_macro, best_p

    def calibrate(self, records: Sequence[TrainingRecord]) -> None:
        """Calibrate (eta, T, T_diff, gw) by macro-accuracy grid on cal split.

        Matches reference v1.0 implementation. up
        to (and including) the final per-difficulty matrix rebuild with the
        winning eta. Selective theta calibration is layered on top by
        :class:`DSNBFSelective.calibrate`.
        """
        if self._train_cache is None:
            raise RuntimeError(
                f"{type(self).__name__}.calibrate called before fit()"
            )
        cal_instances = _records_to_instances(records)

        eta_grid = [3.0, 5.0, 8.0, 15.0]
        best_overall = -1.0
        best_all: tuple[float, float, float, float] = (1.0, 1.0, 0.5, 5.0)

        for eta in eta_grid:
            for d in _DIFFS:
                di = [i for i in self._train_cache if i["difficulty"] == d]
                self._emit_d[d], self._prior_d[d] = _build_hier(
                    di, self._emit_g, eta,
                )
            precomp = self._precompute_cal(cal_instances)
            macro, params = self._grid_search(precomp)
            if macro > best_overall:
                best_overall = macro
                best_all = (*params, eta)

        T, Td, gw, eta = best_all
        self._T = T
        self._T_diff = Td
        self._gw = gw
        self._alpha_d = eta
        for d in _DIFFS:
            di = [i for i in self._train_cache if i["difficulty"] == d]
            self._emit_d[d], self._prior_d[d] = _build_hier(
                di, self._emit_g, eta,
            )
        # Invalidate persona cache (matrices changed).
        self._cached_persona = None
        self._diff_probs = None


# ── DSNBFSelective: SKIP variant with calibrated theta + hyperparams ──────


class DSNBFSelective(DSNBF):
    """DSNBF with calibrated SKIP (margin between top-1 and top-2 < theta).

    SKIP is gated on having at least 2 active sources (single-active-source
    cases always answer, mirroring legacy behaviour). The calibration grid
    matches the legacy ``DiffStratNBF.calibrate`` (eta × Td × T × gw, then
    a single sweep over theta).
    """

    name = "DSNBFSelective"
    requires_fit = True
    requires_calibration = True

    # ---- predict (overrides parent to add SKIP rule) ----

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        self._ensure_persona_prepared(atom)
        mu_q = atom_to_mu_q(atom, qid)
        if not self._emit_g:
            raise RuntimeError(
                "DSNBFSelective.predict_one called before fit()"
            )
        combined = self._combined_probs(qid, mu_q)
        best = max(combined, key=combined.get)  # type: ignore[arg-type]

        n_act = sum(1 for s in SOURCE_NAMES if mu_q.get(s) is not None)
        if n_act >= 2:
            vals = sorted(combined.values(), reverse=True)
            margin = vals[0] - vals[1] if len(vals) >= 2 else 1.0
            if margin < self._theta:
                return Prediction(answer=SKIP_SENTINEL, would_skip=True)
        return Prediction(answer=best, would_skip=False)

    # ---- calibrate ----

    def calibrate(self, records: Sequence[TrainingRecord]) -> None:
        """Calibrate (eta, T, T_diff, gw) via the inherited grid, then theta.

        The (T, Td, gw, eta) sweep is delegated to :meth:`DSNBF.calibrate`;
        only the F_0.5 SKIP-margin theta sweep is added here.
        """
        super().calibrate(records)
        cal_instances = _records_to_instances(records)

        # ── theta search on calibrated hyperparameters ──
        precomp = self._precompute_cal(cal_instances)
        all_dp: list[dict[str, float]] = []
        for pc in precomp:
            dlps: dict[str, float] = {}
            for d in _DIFFS:
                lp = self._diff_log_prior[d]
                for qid in QUESTIONS:
                    ans = QUESTIONS[qid]["answer_space"]
                    terms = [
                        self._prior_d[d][qid][w]
                        + self._T_diff * pc["ed"][qid][d][w]
                        for w in ans
                    ]
                    mx = max(terms)
                    lp += mx + math.log(
                        sum(math.exp(t - mx) for t in terms)
                    )
                dlps[d] = lp
            mx = max(dlps.values())
            raw = {d: math.exp(v - mx) for d, v in dlps.items()}
            st = sum(raw.values())
            all_dp.append({d: p / st for d, p in raw.items()})

        records_obs: list[tuple[float, bool, int]] = []
        for idx, pc in enumerate(precomp):
            ddp = all_dp[idx]
            for qid in QUESTIONS:
                ans = QUESTIONS[qid]["answer_space"]
                lpg = {
                    w: self._prior_g[qid][w] + self._T * pc["eg"][qid][w]
                    for w in ans
                }
                mxg = max(lpg.values())
                prg = {w: math.exp(lpg[w] - mxg) for w in ans}
                stg = sum(prg.values())
                pg = {w: prg[w] / stg for w in ans}
                dwq: dict[str, float] = {w: 0.0 for w in ans}
                for d in _DIFFS:
                    lpd = {
                        w: self._prior_d[d][qid][w]
                        + self._T * pc["ed"][qid][d][w]
                        for w in ans
                    }
                    mxd = max(lpd.values())
                    prd = {w: math.exp(lpd[w] - mxd) for w in ans}
                    std_ = sum(prd.values())
                    for w in ans:
                        dwq[w] += ddp[d] * prd[w] / std_
                cb = {
                    w: self._gw * pg[w] + (1.0 - self._gw) * dwq[w]
                    for w in ans
                }
                vals = sorted(cb.values(), reverse=True)
                margin = vals[0] - vals[1] if len(vals) >= 2 else 1.0
                pred = max(cb, key=cb.get)  # type: ignore[arg-type]
                correct = pred == pc["gt"][qid]
                records_obs.append((margin, correct, pc["n_act"][qid]))

        # F_0.5 grid over theta in {0.00, 0.01, ..., 1.00} — exact match to legacy.
        best_f05 = -1.0
        best_theta = 0.0
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
                best_theta = theta
        self._theta = best_theta
