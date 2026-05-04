"""Reproduce paper DGP Perturbation — DGP perturbation grid (4-seed mean ± σ).

Source: ``Appendix robustness section`` (lines 87-99,
section §C-4) plus four ``data/benchmark/results/robustness_*.json`` files
(one per canonical seed). The robustness experiment sweeps a 3×3 grid
of bias-amplification (b ∈ {0.5, 1.0, 2.0}) × dropout-rate scaling
(d ∈ {0.5, 1.0, 2.0}) applied to the source-projection generators,
producing 9 variants per seed.

9 variants × 7 methods × {mean, σ} = 63 paper-lock cell checks
(each cell asserts BOTH mean and σ within tolerance) + 11 narrative
locks (5 pp range/delta + Kendall τ mean + DSNBF wins + 2 τ counts +
2 bootstrap CI bounds) = 74 total checks.

Method-label → JSON-key (verified empirically against PAPER_TAB_DGP_PERTURBATION):
  DSNBF  → DSNBF-NoSkip   NBF    → NBF-NoSkip   ABF → PRISM-NoSkip
  BCF    → BCF(4p)        MV     → Majority-Vote
  SSB    → SSB-Global     ← NOT Single-Source-Best (paper DGP Perturbation uses
                            the global variant; intentionally diverges
                            from the Forced-Accuracy Main Table/C6 SSB convention which uses
                            Single-Source-Best). Verified empirically against PAPER_TAB_DGP_PERTURBATION.
  Ref. → Oracle-Ext (JSON key)

The Kendall τ claim (paper line ~110, ≈0.77 across 36 variant-pair
ranking comparisons over the canonical 7-method set) and the DSNBF
strict-wins-over-fusion-5 count (32/36 cells) are both locked here.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
from itertools import combinations
from pathlib import Path

from scipy.stats import kendalltau

from .._common import OUTPUT_DIR
from survey2agent._paths import RESULTS_ROOT


# ── Source files & schema ─────────────────────────────────────────────

_RESULTS_DIR: Path = RESULTS_ROOT

SEEDS: tuple[str, ...] = ("s20260321", "s20260322", "s20260323", "s20260324")

# Seed s20260321 lives in the legacy un-suffixed file; the other three
# follow the ``robustness_<seed>.json`` convention.
FILE_MAP: dict[str, Path] = {
    "s20260321": _RESULTS_DIR / "robustness_results.json",
    "s20260322": _RESULTS_DIR / "robustness_s20260322.json",
    "s20260323": _RESULTS_DIR / "robustness_s20260323.json",
    "s20260324": _RESULTS_DIR / "robustness_s20260324.json",
}

# Paper-label → JSON-key (LOCAL map; do NOT reuse PAPER_LABEL_TO_JSON_KEY
# from _appendix_helpers because that maps SSB → Single-Source-Best,
# which is the wrong source for paper DGP Perturbation. See module docstring.)
METHOD_MAP: dict[str, str] = {
    "DSNBF":  "DSNBF-NoSkip",
    "NBF":    "NBF-NoSkip",
    "ABF":    "PRISM-NoSkip",
    "BCF":    "BCF(4p)",
    "MV":     "Majority-Vote",
    "SSB":    "SSB-Global",
    "Ref.": "Oracle-Ext",
}

VARIANTS: tuple[str, ...] = (
    "b0.5_d0.5", "b0.5_d1.0", "b0.5_d2.0",
    "b1.0_d0.5", "b1.0_d1.0", "b1.0_d2.0",
    "b2.0_d0.5", "b2.0_d1.0", "b2.0_d2.0",
)
COL_ORDER: tuple[str, ...] = ("DSNBF", "NBF", "ABF", "BCF", "MV", "SSB", "Ref.")
DISPLAY_LABEL: dict[str, str] = {"SSB": "SSB-G"}


# ── Paper-locked values (mean_pp, sigma_pp) per (variant, method) ─────

PAPER_TAB_DGP_PERTURBATION: dict[str, dict[str, tuple[float, float]]] = {
    "b0.5_d0.5": {"DSNBF": (84.4, 0.9), "NBF": (83.7, 0.8), "ABF": (74.5, 0.3), "BCF": (71.6, 0.7), "MV": (73.0, 0.7), "SSB": (76.2, 0.7), "Ref.": (93.7, 0.3)},
    "b0.5_d1.0": {"DSNBF": (83.1, 1.2), "NBF": (82.7, 0.8), "ABF": (74.5, 1.0), "BCF": (70.2, 0.7), "MV": (72.4, 0.7), "SSB": (77.4, 0.8), "Ref.": (94.0, 0.3)},
    "b0.5_d2.0": {"DSNBF": (83.6, 1.2), "NBF": (83.0, 0.9), "ABF": (73.9, 0.9), "BCF": (68.3, 1.0), "MV": (71.5, 0.9), "SSB": (78.6, 0.9), "Ref.": (94.1, 0.5)},
    "b1.0_d0.5": {"DSNBF": (84.2, 1.0), "NBF": (83.7, 0.8), "ABF": (74.3, 0.8), "BCF": (71.3, 0.7), "MV": (69.8, 0.3), "SSB": (68.9, 0.8), "Ref.": (92.9, 0.4)},
    "b1.0_d1.0": {"DSNBF": (82.3, 0.6), "NBF": (82.0, 0.7), "ABF": (72.0, 0.8), "BCF": (69.8, 0.8), "MV": (69.5, 0.6), "SSB": (70.1, 0.9), "Ref.": (93.2, 0.7)},
    "b1.0_d2.0": {"DSNBF": (81.4, 0.5), "NBF": (80.6, 0.6), "ABF": (70.5, 0.8), "BCF": (68.0, 0.4), "MV": (68.6, 0.6), "SSB": (71.4, 0.7), "Ref.": (92.8, 0.4)},
    "b2.0_d0.5": {"DSNBF": (84.0, 0.5), "NBF": (83.0, 0.5), "ABF": (71.3, 0.7), "BCF": (70.6, 0.9), "MV": (63.4, 0.2), "SSB": (57.3, 1.3), "Ref.": (91.5, 0.4)},
    "b2.0_d1.0": {"DSNBF": (82.4, 1.1), "NBF": (80.7, 1.0), "ABF": (70.3, 0.5), "BCF": (69.4, 0.8), "MV": (63.0, 0.4), "SSB": (58.5, 1.2), "Ref.": (91.2, 0.5)},
    "b2.0_d2.0": {"DSNBF": (80.8, 0.7), "NBF": (79.1, 0.9), "ABF": (68.6, 0.7), "BCF": (67.9, 0.6), "MV": (62.6, 0.5), "SSB": (59.7, 1.1), "Ref.": (90.5, 0.3)},
}

# Narrative-claim locks (paper §C-4 prose, ~lines 110-117). Values in
# percentage points (pp) except Kendall τ (unitless) and DSNBF wins
# (integer count over 36 (seed, variant) cells).
PAPER_NARRATIVE_DGP_PERTURBATION: dict[str, float] = {
    "MV_b2d2_minus_nominal_pp":    -6.9,
    "DSNBF_b2d2_minus_nominal_pp": -1.6,
    "SSB_range_pp":                21.3,
    "DSNBF_range_pp":               3.7,
    "NBF_range_pp":                 4.6,
    "kendall_tau":                  0.77,    # canonical 7-method set, 4-seed × 36 pairs
    "dsnbf_wins_36":               32.0,     # /36 (seed, variant) cells, vs fusion 5 (excl Oracle)
    "kendall_tau_positive_count":  144.0,    # /144 pairs with τ > 0
    "kendall_tau_above_half_count": 143.0,   # /144 pairs with τ > 0.5
    "kendall_tau_ci_lower":         0.74,    # bootstrap 95% CI lower (10000 resamples, seed=42)
    "kendall_tau_ci_upper":         0.80,    # bootstrap 95% CI upper
}

# Tolerances (all in pp on the percent scale unless noted).
TOL_ACC: float = 0.5     # cell mean
TOL_SIGMA: float = 0.5   # cell σ
TOL_NARR: float = 0.3    # narrative range / delta claims (pp)
TOL_TAU: float = 0.02    # Kendall τ (unitless; paper rounds to 1 dp)
TOL_TAU_CI: float = 0.01 # Kendall τ CI bound (paper rounds to 2 dp)
TOL_WINS: float = 0.0    # DSNBF wins / τ counts (exact integer match)

# Bootstrap settings for Kendall τ mean CI.
_BOOTSTRAP_B: int = 10000
_BOOTSTRAP_SEED: int = 42

# Canonical fusion-5 set (excl Oracle) used for DSNBF strict-wins count.
_FUSION5_LABELS: tuple[str, ...] = ("NBF", "ABF", "BCF", "MV", "SSB")


# ── Loaders ───────────────────────────────────────────────────────────


def load_data() -> dict[str, dict[str, dict[str, float]]]:
    """Return ``{seed: {variant: {json_method_key: acc_fraction}}}``.

    Validates that all four files exist and that every (variant, method)
    cell required by PAPER_TAB_DGP_PERTURBATION is present in every seed.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for seed, path in FILE_MAP.items():
        if not path.exists():
            raise FileNotFoundError(f"missing robustness file for {seed}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        vr = payload.get("variant_results")
        if not isinstance(vr, dict):
            raise ValueError(f"{path}: missing 'variant_results' dict")
        out[seed] = vr
    # Schema sanity: each seed must have every variant + every method.
    for seed, vr in out.items():
        for variant in VARIANTS:
            if variant not in vr:
                raise KeyError(f"{seed}: variant {variant!r} not in JSON")
            for label, jk in METHOD_MAP.items():
                if jk not in vr[variant]:
                    raise KeyError(
                        f"{seed}/{variant}: missing JSON method key {jk!r} "
                        f"(paper label {label!r})"
                    )
    return out


def compute_cell(
    data: dict[str, dict[str, dict[str, float]]],
    variant: str,
    method_label: str,
) -> tuple[float, float]:
    """Return (mean_pp, sigma_pp) — 4-seed population mean and stddev
    of the (variant, method) accuracy in percentage points."""
    jk = METHOD_MAP[method_label]
    per_seed = [data[seed][variant][jk] * 100.0 for seed in SEEDS]
    return statistics.mean(per_seed), statistics.pstdev(per_seed)


# ── Checkers ──────────────────────────────────────────────────────────


def check_cells(
    data: dict[str, dict[str, dict[str, float]]],
) -> tuple[int, int, list[str], dict[tuple[str, str], tuple[float, float]]]:
    """Verify all 9 × 7 = 63 (variant, method) cells.

    A cell PASSES iff BOTH ``|mean - paper_mean| <= TOL_ACC`` AND
    ``|sigma - paper_sigma| <= TOL_SIGMA``. Returns the empirical mean/σ
    map for downstream use (markdown rendering, narrative checks).
    """
    fails: list[str] = []
    n_pass = 0
    n_total = 0
    empirical: dict[tuple[str, str], tuple[float, float]] = {}
    for variant in VARIANTS:
        for method in COL_ORDER:
            n_total += 1
            mean_pp, sigma_pp = compute_cell(data, variant, method)
            empirical[(variant, method)] = (mean_pp, sigma_pp)
            paper_mean, paper_sigma = PAPER_TAB_DGP_PERTURBATION[variant][method]
            d_mean = abs(mean_pp - paper_mean)
            d_sigma = abs(sigma_pp - paper_sigma)
            if d_mean <= TOL_ACC and d_sigma <= TOL_SIGMA:
                n_pass += 1
            else:
                display_method = DISPLAY_LABEL.get(method, method)
                fails.append(
                    f"cell {variant}/{display_method}: empirical={mean_pp:.2f}±{sigma_pp:.2f} "
                    f"paper={paper_mean:.1f}±{paper_sigma:.1f} "
                    f"Δmean={d_mean:.2f}pp Δσ={d_sigma:.2f}pp"
                )
    return n_pass, n_total, fails, empirical


def _compute_kendall_taus(
    data: dict[str, dict[str, dict[str, float]]],
) -> list[float]:
    """Return the full list of pairwise Kendall τ values (4 seeds × 36
    variant-pairs = 144 τs) over the canonical 7-method set.

    Per-seed reduction: for each seed, compute τ between rankings of
    the canonical 7-method set under each (variant_i, variant_j) pair.
    Skips NaN τ from tied rankings (defensive; no ties expected on
    these JSON values).
    """
    methods_jk = [METHOD_MAP[m] for m in COL_ORDER]   # 7 keys, fixed order
    taus: list[float] = []
    for seed in SEEDS:
        for v1, v2 in combinations(VARIANTS, 2):
            r1 = [data[seed][v1][jk] for jk in methods_jk]
            r2 = [data[seed][v2][jk] for jk in methods_jk]
            tau, _ = kendalltau(r1, r2)
            if math.isfinite(tau):
                taus.append(tau)
    if not taus:
        raise RuntimeError("Kendall τ computation produced no finite values")
    return taus


def _bootstrap_mean_ci(
    values: list[float],
    n_resamples: int = _BOOTSTRAP_B,
    seed: int = _BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Uses ``random.choices`` with the given seed for reproducibility;
    matches the figures quoted in paper §C-4 (10000 resamples, seed=42).
    """
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = rng.choices(values, k=n)
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples)]
    return lo, hi


def _count_dsnbf_strict_wins(
    data: dict[str, dict[str, dict[str, float]]],
) -> int:
    """Count (seed, variant) cells where DSNBF strictly beats every
    method in the canonical fusion-5 set (NBF, ABF, BCF, MV, SSB).
    """
    dsnbf_jk = METHOD_MAP["DSNBF"]
    others_jk = [METHOD_MAP[m] for m in _FUSION5_LABELS]
    wins = 0
    for seed in SEEDS:
        for variant in VARIANTS:
            d = data[seed][variant][dsnbf_jk]
            if all(d > data[seed][variant][jk] for jk in others_jk):
                wins += 1
    return wins


def check_narrative(
    empirical: dict[tuple[str, str], tuple[float, float]],
    data: dict[str, dict[str, dict[str, float]]],
) -> tuple[int, int, list[str]]:
    """Verify the 11 §C-4 narrative claims (5 pp + Kendall τ mean +
    DSNBF wins + 2 τ counts + 2 bootstrap CI bounds)."""
    fails: list[str] = []

    def _means_for(method: str) -> list[float]:
        return [empirical[(v, method)][0] for v in VARIANTS]

    nominal = "b1.0_d1.0"
    extreme = "b2.0_d2.0"
    claims: dict[str, float] = {}

    claims["MV_b2d2_minus_nominal_pp"] = (
        empirical[(extreme, "MV")][0] - empirical[(nominal, "MV")][0]
    )
    claims["DSNBF_b2d2_minus_nominal_pp"] = (
        empirical[(extreme, "DSNBF")][0] - empirical[(nominal, "DSNBF")][0]
    )
    ssb_means = _means_for("SSB")
    claims["SSB_range_pp"] = max(ssb_means) - min(ssb_means)
    dsnbf_means = _means_for("DSNBF")
    claims["DSNBF_range_pp"] = max(dsnbf_means) - min(dsnbf_means)
    nbf_means = _means_for("NBF")
    claims["NBF_range_pp"] = max(nbf_means) - min(nbf_means)

    taus = _compute_kendall_taus(data)
    claims["kendall_tau"] = sum(taus) / len(taus)
    claims["dsnbf_wins_36"] = float(_count_dsnbf_strict_wins(data))
    claims["kendall_tau_positive_count"] = float(sum(1 for t in taus if t > 0))
    claims["kendall_tau_above_half_count"] = float(sum(1 for t in taus if t > 0.5))
    ci_lo, ci_hi = _bootstrap_mean_ci(taus)
    claims["kendall_tau_ci_lower"] = ci_lo
    claims["kendall_tau_ci_upper"] = ci_hi

    _COUNT_CLAIMS = {
        "dsnbf_wins_36",
        "kendall_tau_positive_count",
        "kendall_tau_above_half_count",
    }
    _CI_CLAIMS = {"kendall_tau_ci_lower", "kendall_tau_ci_upper"}

    n_pass = 0
    for name, paper_v in PAPER_NARRATIVE_DGP_PERTURBATION.items():
        emp = claims[name]
        d = abs(emp - paper_v)
        if name == "kendall_tau":
            tol = TOL_TAU
            unit = ""
        elif name in _CI_CLAIMS:
            tol = TOL_TAU_CI
            unit = ""
        elif name in _COUNT_CLAIMS:
            tol = TOL_WINS
            unit = " cells"
        else:
            tol = TOL_NARR
            unit = "pp"
        if d <= tol:
            n_pass += 1
        else:
            fails.append(
                f"narrative {name}: empirical={emp:+.4f}{unit} paper={paper_v:+.4f}{unit} "
                f"Δ={d:.4f} (tol {tol}{unit})"
            )
    return n_pass, len(PAPER_NARRATIVE_DGP_PERTURBATION), fails


# ── Output (CSV + MD) ─────────────────────────────────────────────────


def _render_outputs(
    empirical: dict[tuple[str, str], tuple[float, float]],
    narrative_pass: int,
    narrative_total: int,
    cell_fails: list[str],
    narrative_fails: list[str],
) -> tuple[Path, Path]:
    """Write a self-contained CSV + Markdown for DGP Perturbation (schema differs
    from the standard CSV_COLUMNS used by other appendix scripts: each
    row has 6 columns capturing both mean and σ side-by-side)."""
    out_dir = OUTPUT_DIR / "appendix"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dgp_perturbation.csv"
    md_path = out_dir / "dgp_perturbation.md"

    csv_lines: list[str] = [
        "# Generated by paper_artifacts.appendix.dgp_perturbation",
        f"# Tolerances: ±{TOL_ACC}pp on cell mean, ±{TOL_SIGMA}pp on cell σ, "
        f"±{TOL_NARR}pp on narrative claims",
        "variant,method,empirical_mean_pp,empirical_sigma_pp,"
        "paper_mean_pp,paper_sigma_pp,paper_match",
    ]
    for variant in VARIANTS:
        for method in COL_ORDER:
            mean_pp, sigma_pp = empirical[(variant, method)]
            paper_mean, paper_sigma = PAPER_TAB_DGP_PERTURBATION[variant][method]
            display_method = DISPLAY_LABEL.get(method, method)
            ok = (
                abs(mean_pp - paper_mean) <= TOL_ACC
                and abs(sigma_pp - paper_sigma) <= TOL_SIGMA
            )
            csv_lines.append(
                f"{variant},{display_method},{mean_pp:.2f},{sigma_pp:.2f},"
                f"{paper_mean:.1f},{paper_sigma:.1f},{'OK' if ok else 'FAIL'}"
            )
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md_lines: list[str] = [
        "# dgp_perturbation",
        "",
        "**DGP Perturbation.** DGP perturbation grid: 4-seed mean ± σ macro accuracy "
        "(%) under bias-amplification × dropout-scaling sweeps. Each cell "
        "shows the empirical (mean ± σ) value; paper values are within "
        f"±{TOL_ACC} pp on mean and ±{TOL_SIGMA} pp on σ for all 63 cells.",
        "",
        "| variant | " + " | ".join(DISPLAY_LABEL.get(m, m) for m in COL_ORDER) + " |",
        "|:---|" + "|".join([":---:"] * len(COL_ORDER)) + "|",
    ]
    for variant in VARIANTS:
        cells = []
        for method in COL_ORDER:
            mean_pp, sigma_pp = empirical[(variant, method)]
            cells.append(f"{mean_pp:.1f}±{sigma_pp:.1f}")
        md_lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    md_lines += [
        "",
        f"*Narrative claims locked: {narrative_pass}/{narrative_total} within "
        f"±{TOL_NARR} pp.*",
        "*Source: `data/benchmark/results/robustness_*.json` (one per seed). "
        "Paper location: `Appendix robustness section` "
        "(`tab:F4`, §C-4).*",
    ]
    if cell_fails or narrative_fails:
        md_lines.append("")
        md_lines.append("### Failures")
        for msg in cell_fails + narrative_fails:
            md_lines.append(f"- {msg}")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return csv_path, md_path


# ── Entry point ───────────────────────────────────────────────────────


def main() -> int:
    t0 = time.time()
    print("[dgp_perturbation] loading robustness JSONs...", flush=True)
    data = load_data()

    cell_pass, cell_total, cell_fails, empirical = check_cells(data)
    narr_pass, narr_total, narr_fails = check_narrative(empirical, data)

    csv_path, md_path = _render_outputs(
        empirical, narr_pass, narr_total, cell_fails, narr_fails
    )

    total_pass = cell_pass + narr_pass
    total = cell_total + narr_total
    elapsed = time.time() - t0
    print(f"\n=== DGP Perturbation DGP Perturbation ===")
    print(f"Cells:     {cell_pass}/{cell_total}")
    print(f"Narrative: {narr_pass}/{narr_total}")
    print(
        f"TOTAL:     {total_pass}/{total} "
        f"({total_pass * 100 / total:.1f}%) in {elapsed:.1f}s"
    )
    print(f"Outputs:   {csv_path.name}, {md_path.name}")
    for msg in cell_fails + narr_fails:
        print(f"  FAIL: {msg}")
    return 0 if total_pass == total else (total - total_pass)


if __name__ == "__main__":
    sys.exit(main())
