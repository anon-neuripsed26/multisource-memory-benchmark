"""Reproduce paper Atom Extraction Faithfulness Audit — Atom extraction faithfulness audit.

Source: ``Appendix benchmark-details section``
(``\\subsection{Atom Extraction Faithfulness Audit}\\label{app:audit}``,
lines ~278-301), with the load-bearing 93.08% headline echoed inline at
line 204 (Stage-3 walked-through example) and used to support §s4's
"81% method / 19% input" factorial framing.

Audit definition (paper text)
-----------------------------
At the ``(persona, seed, question, source)`` cell level on the full
test split (120 personas x 4 seeds x 18 questions x 5 sources = 43,200
cells), compare:

  - mu_hat: GPT-5.4 LLM-extracted atom (from NL memory render)
  - mu_star: deterministic Python read-out from the structured L_3 JSON

Cells where both mu_hat and mu_star are ``None`` count as equal (the
schema-empty case, primarily ``device_log`` for non-exercise questions
and ``objective_log`` outside its observation domain).

Paper-lock cells (Atom Extraction Faithfulness Audit body): 6 source rows (5 per-source + Overall),
each with one P(mu_hat = mu_star) percentage.

Paper-lock narrative claims (B-4 prose):

  - Overall faithfulness: 93.08%  (also echoed at L204; load-bearing for s4)
  - Per-source max:       96.28%  (planner)
  - Per-source min:       86.46%  (objective_log)
  - Total cell count:     43,200  (sanity: 120 x 4 x 18 x 5)
  - Per-seed range low:   92.66%
  - Per-seed range high:  93.94%
  - Stable difficulty:    93.02%
  - Temporal-shift:       93.81%
  - Stated-vs-revealed:   92.40%
  - Per-Q bottom-3 (A2, G2, F2 in the paper text, equality 81-84%):
      * lowest (F2):       81.33%   (paper "81%")
      * highest of three (A2): 83.63%  (paper "84%")

Tolerances
----------
  - Cells (5 per-source + Overall): +/- 0.05 pp (paper reports 2 dp).
  - Narrative %s reported with 2 dp:  +/- 0.05 pp.
  - Per-Q "81-84%" range bound (1 dp): +/- 0.5 pp.
  - Cell count: exact.

The Atom Extraction Faithfulness Audit body in the paper has no ``\\label{}`` (verified by
``grep -n "label{tab" appendix_b_benchmark_details.tex``); we use the
internal key ``Atom Extraction Faithfulness Audit`` / ``b4_atom_faithfulness`` for paper-artifacts
bookkeeping. Adding an explicit LaTeX label is left to a separate paper
edit (out of scope for this lock).

Reproduces against the frozen extracted atoms in
``data/extracted_atoms/{seed}/`` (mu_hat, GPT-5.4 NL
extraction) and the deterministic oracle atoms via
``load_atoms_for_seed(seed, mode="oracle")`` (mu_star).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from survey2agent.evaluation.data_loaders import (
    load_atoms_for_seed,
    load_persona_difficulty_index,
    load_splits,
)
from survey2agent.extraction.atoms import (
    EXPECTED_QUESTION_IDS,
    EXPECTED_SOURCES,
)

from .._common import OUTPUT_DIR


SEEDS: tuple[str, ...] = ("s20260321", "s20260322", "s20260323", "s20260324")
DIFFICULTIES: tuple[str, ...] = ("stable", "temporal_shift", "stated_vs_revealed")

# Paper Atom Extraction Faithfulness Audit body: per-source equality rates (percent).
PAPER_TAB_ATOM_FAITHFULNESS: dict[str, float] = {
    "profile_ltm":        91.24,
    "planner":            96.28,
    "daily_self_report":  96.20,
    "objective_log":      86.46,
    "device_log":         95.20,
    "Overall":            93.08,
}

# Paper §B-4 prose narrative claims (also: 93.08% repeated inline at L204).
PAPER_NARRATIVE_ATOM_FAITHFULNESS: dict[str, float] = {
    "overall_faithfulness_pct":     93.08,
    "per_source_max_pct":           96.28,
    "per_source_min_pct":           86.46,
    "total_cells":                  43200.0,   # 120 x 4 x 18 x 5
    "per_seed_min_pct":             92.66,
    "per_seed_max_pct":             93.94,
    "diff_stable_pct":              93.02,
    "diff_temporal_shift_pct":      93.81,
    "diff_stated_vs_revealed_pct":  92.40,
    "per_q_bottom_pct":             81.33,     # F2; paper "81%"
    "per_q_bottom3_top_pct":        83.63,     # A2; paper "84%"
}

# Tolerances (pp on the percent scale; cell count exact).
TOL_CELL: float = 0.05
TOL_NARR_PCT: float = 0.05
TOL_NARR_PCT_1DP: float = 0.5  # for the "81-84%" bound
TOL_COUNT: float = 0.0

# Map narrative key -> tolerance (pp; cell count uses TOL_COUNT).
_NARR_TOLERANCE: dict[str, float] = {
    "overall_faithfulness_pct":    TOL_NARR_PCT,
    "per_source_max_pct":          TOL_NARR_PCT,
    "per_source_min_pct":          TOL_NARR_PCT,
    "total_cells":                 TOL_COUNT,
    "per_seed_min_pct":            TOL_NARR_PCT,
    "per_seed_max_pct":            TOL_NARR_PCT,
    "diff_stable_pct":             TOL_NARR_PCT,
    "diff_temporal_shift_pct":     TOL_NARR_PCT,
    "diff_stated_vs_revealed_pct": TOL_NARR_PCT,
    "per_q_bottom_pct":            TOL_NARR_PCT_1DP,
    "per_q_bottom3_top_pct":       TOL_NARR_PCT_1DP,
}


# ── Computation ───────────────────────────────────────────────────────────


def compute_atom_faithfulness() -> dict[str, dict[str, tuple[int, int]]]:
    """Compute (eq_count, total_count) over the test split, broken down
    by source / seed / difficulty / question.

    Returns a dict with keys ``per_source``, ``per_seed``, ``per_diff``,
    ``per_question``, ``overall``; each value maps a label to a
    ``(equal_cells, total_cells)`` pair.

    Cells where ``mu_hat`` and ``mu_star`` are both ``None`` count as
    equal (consistent with paper §B-4 footer: "Cells where both
    mu_hat and mu_star are null count as equal").
    """
    splits = load_splits()
    test_personas = sorted(splits["test"])
    diff_index = load_persona_difficulty_index()

    per_source: dict[str, list[int]] = {s: [0, 0] for s in EXPECTED_SOURCES}
    per_seed: dict[str, list[int]] = {s: [0, 0] for s in SEEDS}
    per_diff: dict[str, list[int]] = {d: [0, 0] for d in DIFFICULTIES}
    per_question: dict[str, list[int]] = {q: [0, 0] for q in EXPECTED_QUESTION_IDS}
    overall = [0, 0]

    for seed in SEEDS:
        llm = load_atoms_for_seed(seed, mode="llm")
        oracle = load_atoms_for_seed(seed, mode="oracle")
        for pid in test_personas:
            d = diff_index[pid]
            l_ext = llm[pid].extraction
            o_ext = oracle[pid].extraction
            for qid in EXPECTED_QUESTION_IDS:
                for src in EXPECTED_SOURCES:
                    lv = l_ext[qid][src]
                    ov = o_ext[qid][src]
                    same = (lv == ov)
                    overall[1] += 1
                    per_source[src][1] += 1
                    per_seed[seed][1] += 1
                    per_diff[d][1] += 1
                    per_question[qid][1] += 1
                    if same:
                        overall[0] += 1
                        per_source[src][0] += 1
                        per_seed[seed][0] += 1
                        per_diff[d][0] += 1
                        per_question[qid][0] += 1

    return {
        "per_source":   {k: (v[0], v[1]) for k, v in per_source.items()},
        "per_seed":     {k: (v[0], v[1]) for k, v in per_seed.items()},
        "per_diff":     {k: (v[0], v[1]) for k, v in per_diff.items()},
        "per_question": {k: (v[0], v[1]) for k, v in per_question.items()},
        "overall":      {"Overall": (overall[0], overall[1])},
    }


def _pct(eq: int, tot: int) -> float:
    return 100.0 * eq / tot if tot else 0.0


# ── Checkers ──────────────────────────────────────────────────────────────


def check_cells(
    counts: dict[str, dict[str, tuple[int, int]]],
) -> tuple[int, int, list[str], dict[str, float]]:
    """Verify the 6 Atom Extraction Faithfulness Audit body cells (5 per-source + Overall)."""
    fails: list[str] = []
    n_pass = 0
    n_total = 0
    empirical: dict[str, float] = {}
    cell_order: list[str] = list(EXPECTED_SOURCES) + ["Overall"]
    for label in cell_order:
        if label == "Overall":
            eq, tot = counts["overall"]["Overall"]
        else:
            eq, tot = counts["per_source"][label]
        emp_pct = _pct(eq, tot)
        empirical[label] = emp_pct
        paper_pct = PAPER_TAB_ATOM_FAITHFULNESS[label]
        d = abs(emp_pct - paper_pct)
        n_total += 1
        if d <= TOL_CELL:
            n_pass += 1
        else:
            fails.append(
                f"cell {label}: empirical={emp_pct:.4f}% paper={paper_pct:.2f}% "
                f"delta={d:.4f}pp (tol {TOL_CELL}pp)"
            )
    return n_pass, n_total, fails, empirical


def check_narrative(
    counts: dict[str, dict[str, tuple[int, int]]],
) -> tuple[int, int, list[str], dict[str, float]]:
    """Verify the 11 §B-4 narrative claims."""
    fails: list[str] = []
    claims: dict[str, float] = {}

    overall_eq, overall_tot = counts["overall"]["Overall"]
    claims["overall_faithfulness_pct"] = _pct(overall_eq, overall_tot)
    claims["total_cells"] = float(overall_tot)

    per_source_pcts = {
        s: _pct(eq, tot) for s, (eq, tot) in counts["per_source"].items()
    }
    claims["per_source_max_pct"] = max(per_source_pcts.values())
    claims["per_source_min_pct"] = min(per_source_pcts.values())

    per_seed_pcts = [
        _pct(eq, tot) for (eq, tot) in counts["per_seed"].values()
    ]
    claims["per_seed_min_pct"] = min(per_seed_pcts)
    claims["per_seed_max_pct"] = max(per_seed_pcts)

    pd = counts["per_diff"]
    claims["diff_stable_pct"] = _pct(*pd["stable"])
    claims["diff_temporal_shift_pct"] = _pct(*pd["temporal_shift"])
    claims["diff_stated_vs_revealed_pct"] = _pct(*pd["stated_vs_revealed"])

    # Per-Q "A2, G2, F2 81-84%" bottom-3 bracket.
    per_q_pcts: list[tuple[str, float]] = sorted(
        ((q, _pct(eq, tot)) for q, (eq, tot) in counts["per_question"].items()),
        key=lambda kv: kv[1],
    )
    bottom3 = per_q_pcts[:3]
    bottom3_qs = {q for q, _ in bottom3}
    if bottom3_qs != {"A2", "G2", "F2"}:
        fails.append(
            f"per-Q bottom-3 mismatch: empirical={sorted(bottom3_qs)} "
            f"paper={{A2,G2,F2}}"
        )
    claims["per_q_bottom_pct"] = bottom3[0][1]
    claims["per_q_bottom3_top_pct"] = bottom3[2][1]

    n_total = len(PAPER_NARRATIVE_ATOM_FAITHFULNESS)
    n_pass = 0
    for name, paper_v in PAPER_NARRATIVE_ATOM_FAITHFULNESS.items():
        emp = claims[name]
        tol = _NARR_TOLERANCE[name]
        d = abs(emp - paper_v)
        if d <= tol:
            n_pass += 1
        else:
            unit = "" if name == "total_cells" else "pp"
            fails.append(
                f"narrative {name}: empirical={emp:.4f}{unit} "
                f"paper={paper_v:.4f}{unit} delta={d:.4f} (tol {tol}{unit})"
            )
    return n_pass, n_total, fails, claims


# ── Output ────────────────────────────────────────────────────────────────


def _render_outputs(
    cell_empirical: dict[str, float],
    narrative_claims: dict[str, float],
    cell_pass: int,
    cell_total: int,
    narr_pass: int,
    narr_total: int,
    cell_fails: list[str],
    narr_fails: list[str],
) -> tuple[Path, Path]:
    out_dir = OUTPUT_DIR / "appendix"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "atom_extraction_faithfulness.csv"
    md_path = out_dir / "atom_extraction_faithfulness.md"

    cell_order: list[str] = list(EXPECTED_SOURCES) + ["Overall"]
    csv_lines: list[str] = [
        "# Generated by paper_artifacts.appendix.atom_extraction_faithfulness",
        f"# Cell tolerance: +/-{TOL_CELL}pp; narrative tolerance: "
        f"+/-{TOL_NARR_PCT}pp (2dp claims), +/-{TOL_NARR_PCT_1DP}pp (1dp claims)",
        "kind,label,empirical_pct,paper_value,paper_match",
    ]
    for label in cell_order:
        emp = cell_empirical[label]
        pap = PAPER_TAB_ATOM_FAITHFULNESS[label]
        ok = abs(emp - pap) <= TOL_CELL
        csv_lines.append(
            f"cell,{label},{emp:.4f},{pap:.2f},{'OK' if ok else 'FAIL'}"
        )
    for name, paper_v in PAPER_NARRATIVE_ATOM_FAITHFULNESS.items():
        emp = narrative_claims[name]
        tol = _NARR_TOLERANCE[name]
        ok = abs(emp - paper_v) <= tol
        csv_lines.append(
            f"narrative,{name},{emp:.4f},{paper_v:.4f},{'OK' if ok else 'FAIL'}"
        )
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md_lines: list[str] = [
        "# atom_extraction_faithfulness",
        "",
        "**Atom Extraction Faithfulness Audit.** Atom extraction faithfulness audit on the full "
        "test split (120 personas x 4 seeds x 18 questions x 5 sources "
        "= 43,200 cells). P(mu_hat = mu_star) per source, where mu_hat "
        "is the GPT-5.4 NL extraction and mu_star is the deterministic "
        "structured read-out. Cells where both are null count as equal.",
        "",
        "| Source | P(mu_hat = mu_star) | Paper |",
        "|:---|---:|---:|",
    ]
    for label in EXPECTED_SOURCES:
        md_lines.append(
            f"| `{label}` | {cell_empirical[label]:.2f}% | "
            f"{PAPER_TAB_ATOM_FAITHFULNESS[label]:.2f}% |"
        )
    md_lines.append(
        f"| **Overall** | **{cell_empirical['Overall']:.2f}%** | "
        f"**{PAPER_TAB_ATOM_FAITHFULNESS['Overall']:.2f}%** |"
    )
    md_lines += [
        "",
        f"*Cells locked: {cell_pass}/{cell_total} within +/-{TOL_CELL} pp.*",
        f"*Narrative claims locked: {narr_pass}/{narr_total} "
        f"(2dp claims +/-{TOL_NARR_PCT}pp, 1dp claims +/-{TOL_NARR_PCT_1DP}pp).*",
        "",
        "**Narrative-claim breakdown.**",
        "",
        "| Claim | Empirical | Paper | Delta |",
        "|:---|---:|---:|---:|",
    ]
    for name, paper_v in PAPER_NARRATIVE_ATOM_FAITHFULNESS.items():
        emp = narrative_claims[name]
        d = abs(emp - paper_v)
        unit = "" if name == "total_cells" else "%"
        emp_s = f"{emp:.0f}" if name == "total_cells" else f"{emp:.2f}{unit}"
        pap_s = f"{paper_v:.0f}" if name == "total_cells" else f"{paper_v:.2f}{unit}"
        md_lines.append(f"| {name} | {emp_s} | {pap_s} | {d:.4f} |")
    md_lines += [
        "",
        "*Source: `data/extracted_atoms/{seed}/` (mu_hat, "
        "GPT-5.4 NL extraction) and `$S2A_DATA_ROOT/benchmark/seeds/"
        "{seed}/` (mu_star, deterministic structured read-out via "
        "`load_atoms_for_seed(seed, mode='oracle')`).*",
        "",
        "*Paper location: `Appendix benchmark-details section`, "
        "Atom Extraction Faithfulness Audit; the "
        "93.08% headline is also load-bearing for the s4 \"81% method / "
        "19% input\" factorial framing and is echoed at line 204).*",
    ]
    if cell_fails or narr_fails:
        md_lines.append("")
        md_lines.append("### Failures")
        for msg in cell_fails + narr_fails:
            md_lines.append(f"- {msg}")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return csv_path, md_path


# ── Entry point ───────────────────────────────────────────────────────────


def main() -> int:
    t0 = time.time()
    print("[atom_extraction_faithfulness] computing audit on test split "
          "(120 personas x 4 seeds x 18 q x 5 sources)...", flush=True)
    counts = compute_atom_faithfulness()
    cell_pass, cell_total, cell_fails, cell_empirical = check_cells(counts)
    narr_pass, narr_total, narr_fails, narrative_claims = check_narrative(counts)

    csv_path, md_path = _render_outputs(
        cell_empirical, narrative_claims,
        cell_pass, cell_total, narr_pass, narr_total,
        cell_fails, narr_fails,
    )

    total_pass = cell_pass + narr_pass
    total = cell_total + narr_total
    elapsed = time.time() - t0
    print(f"\n=== Atom Extraction Faithfulness Audit Atom Extraction Faithfulness ===")
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
