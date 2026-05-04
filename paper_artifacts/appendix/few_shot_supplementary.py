"""Reproduce paper Few-Shot Supplementary Check — Few-Shot Supplementary Check.

Source: ``Appendix selective-QA section`` (lines 43-50)
plus ``Experiments section`` ("seed-1 few-shot variant
gains +6.5 pp over GPT Direct yet still falls 4.9 pp below DSNBF").

A single-seed (s20260321), single-model (GPT-5.4), k=3 per-question
exemplar variant. Uses one frozen artifact directory:
``data/method_outputs/gpt-5.4/s20260321/few-shot/{persona}__{qid}.json``
(120 personas × 18 qids = 2160 files). Compared against the matching
GPT-5.4 Direct seed-1 frozen output for the +6.5 pp gain.

Paper-lock cells (11 total):

  * (1)  Few-shot forced accuracy (seed 1)            : 75.4%
  * (2)  GPT-Direct forced accuracy (seed 1)          : 68.9%
  * (3)  Few-shot − Direct gain                       : +6.5 pp
  * (4)  DSNBF forced accuracy (4-seed)               : 80.3%   [paper constant]
  * (5)  Few-shot − DSNBF gap                         : −4.9 pp
  * (6)  Few-shot skip rate                           :  2.6%
  * (7)  Few-shot selective accuracy (would_skip=F)   : 76.1%
  * (8)  Few-shot coverage                            : 97.4%
  * (9)  Forced accuracy on skipped subset            : 46.4%
  * (10) Few-shot per-type F-Missing-Data accuracy    : 63.1%
  * (11) Few-shot per-type G-Annotation accuracy      : 53.8%

DSNBF per-type values (F 77.9 / G 63.5) are paper-constants too — they
belong to ``per_type_accuracy.py``'s reproduction surface, not here. We
only report the *gap* indirectly via the few-shot per-type cells.

Tolerance: ±0.005 absolute on every fraction (≈ ±0.5 pp).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .._common import emit_row, write_outputs, PAPER_TOLERANCE


# Repo root () — script lives at paper_artifacts/appendix/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEED = "s20260321"
_FS_DIR = _REPO_ROOT / "data" / "method_outputs" / "gpt-5.4" / _SEED / "few-shot"
_DIRECT_DIR = _REPO_ROOT / "data" / "method_outputs" / "gpt-5.4" / _SEED / "direct"
_GT_DIR = _REPO_ROOT / "data" / "benchmark" / "seeds" / _SEED


# Paper constants (fractions).
PAPER_FS_FORCED: float = 0.754
PAPER_DIRECT_FORCED: float = 0.689
PAPER_FS_GAIN: float = 0.065        # +6.5 pp
PAPER_DSNBF_4SEED: float = 0.803    # constant (reproduced by per_type_accuracy)
PAPER_FS_VS_DSNBF: float = -0.049   # −4.9 pp
PAPER_FS_SKIP_RATE: float = 0.026
PAPER_FS_SELECTIVE: float = 0.761
PAPER_FS_COVERAGE: float = 0.974
PAPER_FS_SKIPPED_ACC: float = 0.464
PAPER_FS_F_MISSING: float = 0.631
PAPER_FS_G_ANNOT: float = 0.538


# Question → reasoning-type prefix mapping (paper §3 / current question spec).
def _qtype(qid: str) -> str:
    if qid.startswith("Ctrl"):
        return "Ctrl"
    return qid[0]


# Few-shot exemplars cover the 18 canonical benchmark questions.
_FEW_SHOT_QIDS: frozenset[str] = frozenset({
    "A1", "A2", "A3", "B2", "B3", "C2", "C3", "D1", "D2",
    "E1", "E2", "F1", "F2", "F3", "G1", "G2", "Ctrl1", "Ctrl2",
})


def _load_ground_truth() -> dict[str, dict[str, str]]:
    """Return ``{persona_id: {qid: gt_label}}``."""
    out: dict[str, dict[str, str]] = {}
    for persona_dir in _GT_DIR.iterdir():
        if not persona_dir.is_dir():
            continue
        gt_file = persona_dir / "ground_truth.json"
        if not gt_file.exists():
            continue
        gt = json.loads(gt_file.read_text(encoding="utf-8"))
        out[persona_dir.name] = {
            qid: rec.get("answer") for qid, rec in gt.items() if isinstance(rec, dict)
        }
    return out


def _load_few_shot() -> list[dict]:
    """Return a list of few-shot result dicts (each = one (persona, qid) call)."""
    out: list[dict] = []
    for path in sorted(_FS_DIR.glob("*.json")):
        if path.name == "README.md":
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or "persona" not in rec or "question" not in rec:
            continue
        out.append(rec)
    return out


def _load_direct_seed1() -> dict[tuple[str, str], str]:
    """Return ``{(persona, qid): predicted_label}`` for GPT-Direct seed 1."""
    out: dict[tuple[str, str], str] = {}
    for path in sorted(_DIRECT_DIR.glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        persona = rec.get("persona")
        answers = rec.get("answers", {})
        if not isinstance(persona, str) or not isinstance(answers, dict):
            continue
        for qid, ans_rec in answers.items():
            if isinstance(ans_rec, dict):
                pred = ans_rec.get("answer")
                if isinstance(pred, str):
                    out[(persona, qid)] = pred
    return out


def run() -> int:
    gts = _load_ground_truth()
    fs_records = _load_few_shot()
    direct_preds = _load_direct_seed1()

    # ── Few-shot aggregate stats ─────────────────────────────────────────
    fs_total = 0
    fs_correct = 0
    fs_skip_total = 0          # n with would_skip=True
    fs_skip_correct = 0        # would_skip=True ∧ pred==gt (forced acc on skipped)
    fs_kept_total = 0          # would_skip=False
    fs_kept_correct = 0        # selective accuracy numerator
    per_type: dict[str, dict[str, int]] = {}

    for r in fs_records:
        persona = r["persona"]
        qid = r["question"]
        pred = r.get("answer")
        would_skip = bool(r.get("would_skip", False))
        gt = gts.get(persona, {}).get(qid)
        if gt is None or pred is None:
            continue
        # Restrict to the 18 few-shot qids (defensive; legacy artifact may include extras).
        if qid not in _FEW_SHOT_QIDS:
            continue
        is_correct = (pred == gt)
        fs_total += 1
        if is_correct:
            fs_correct += 1
        if would_skip:
            fs_skip_total += 1
            if is_correct:
                fs_skip_correct += 1
        else:
            fs_kept_total += 1
            if is_correct:
                fs_kept_correct += 1
        t = _qtype(qid)
        bucket = per_type.setdefault(t, {"correct": 0, "total": 0})
        bucket["total"] += 1
        if is_correct:
            bucket["correct"] += 1

    # ── GPT-Direct seed-1 forced accuracy on the same (persona, qid) set ─
    # Match the few-shot evaluation set 1:1 so the +6.5 pp gain is apples-to-apples
    # (Direct artifact may contain provider-side extras; few-shot uses the 18 canonical qids).
    direct_total = 0
    direct_correct = 0
    for r in fs_records:
        persona = r["persona"]
        qid = r["question"]
        if qid not in _FEW_SHOT_QIDS:
            continue
        gt = gts.get(persona, {}).get(qid)
        pred = direct_preds.get((persona, qid))
        if gt is None or pred is None:
            continue
        direct_total += 1
        if pred == gt:
            direct_correct += 1

    fs_forced = fs_correct / fs_total if fs_total else 0.0
    fs_skip_rate = fs_skip_total / fs_total if fs_total else 0.0
    fs_selective = fs_kept_correct / fs_kept_total if fs_kept_total else 0.0
    fs_coverage = fs_kept_total / fs_total if fs_total else 0.0
    fs_skipped_acc = fs_skip_correct / fs_skip_total if fs_skip_total else 0.0
    direct_forced = direct_correct / direct_total if direct_total else 0.0
    fs_gain = fs_forced - direct_forced
    fs_vs_dsnbf = fs_forced - PAPER_DSNBF_4SEED

    f_pt = per_type.get("F", {"correct": 0, "total": 0})
    g_pt = per_type.get("G", {"correct": 0, "total": 0})
    fs_f = f_pt["correct"] / f_pt["total"] if f_pt["total"] else 0.0
    fs_g = g_pt["correct"] / g_pt["total"] if g_pt["total"] else 0.0

    # ── Emit rows ────────────────────────────────────────────────────────
    rows: list[dict] = []
    n_personas = len(gts)

    def push(*, row_id: str, label: str, mode: str, metric: str,
             point: float, paper: float) -> None:
        rows.append(emit_row(
            row_id=row_id, method_label=label, mode=mode, metric=metric,
            point=point, n_seeds=1, n_personas=n_personas,
            paper_point=paper, tolerance=PAPER_TOLERANCE,
        ))

    push(row_id="fs_forced",         label="Few-Shot k=3",   mode="seed1",
         metric="forced_accuracy",   point=fs_forced,        paper=PAPER_FS_FORCED)
    push(row_id="direct_forced",     label="GPT-5.4 Direct", mode="seed1",
         metric="forced_accuracy",   point=direct_forced,    paper=PAPER_DIRECT_FORCED)
    push(row_id="fs_vs_direct_gain", label="Few-Shot − Direct", mode="seed1",
         metric="delta_pp",          point=fs_gain,          paper=PAPER_FS_GAIN)
    push(row_id="dsnbf_forced",      label="DSNBF",          mode="4-seed",
         metric="forced_accuracy",   point=PAPER_DSNBF_4SEED, paper=PAPER_DSNBF_4SEED)
    push(row_id="fs_vs_dsnbf_gap",   label="Few-Shot − DSNBF", mode="mixed",
         metric="delta_pp",          point=fs_vs_dsnbf,      paper=PAPER_FS_VS_DSNBF)
    push(row_id="fs_skip_rate",      label="Few-Shot k=3",   mode="seed1",
         metric="skip_rate",         point=fs_skip_rate,     paper=PAPER_FS_SKIP_RATE)
    push(row_id="fs_selective",      label="Few-Shot k=3",   mode="seed1",
         metric="selective_accuracy", point=fs_selective,    paper=PAPER_FS_SELECTIVE)
    push(row_id="fs_coverage",       label="Few-Shot k=3",   mode="seed1",
         metric="coverage",          point=fs_coverage,      paper=PAPER_FS_COVERAGE)
    push(row_id="fs_skipped_acc",    label="Few-Shot k=3",   mode="seed1",
         metric="forced_accuracy_on_skipped", point=fs_skipped_acc,
         paper=PAPER_FS_SKIPPED_ACC)
    push(row_id="fs_per_type_F",     label="Few-Shot k=3",   mode="seed1",
         metric="forced_accuracy_F_Missing_Data", point=fs_f,
         paper=PAPER_FS_F_MISSING)
    push(row_id="fs_per_type_G",     label="Few-Shot k=3",   mode="seed1",
         metric="forced_accuracy_G_Annotation",   point=fs_g,
         paper=PAPER_FS_G_ANNOT)

    # ── Markdown body ────────────────────────────────────────────────────
    md_lines = [
        "| Cell | Method | Metric | Computed | Paper | Match |",
        "|:-----|:-------|:-------|---------:|------:|:-----:|",
    ]
    for row in rows:
        pt = row["point"]
        pv = row["paper_value_point"]
        match = row["paper_match"]
        # Format gain/gap with sign, others as percent.
        if row["metric"] == "delta_pp":
            pt_str = f"{100 * pt:+.1f}"
            pv_str = f"{100 * pv:+.1f}"
        else:
            pt_str = f"{100 * pt:.1f}"
            pv_str = f"{100 * pv:.1f}"
        md_lines.append(
            f"| {row['row_id']} | {row['method_label']} | {row['metric']} "
            f"| {pt_str}% | {pv_str}% | {match} |"
        )
    md_table = "\n".join(md_lines)

    csv_p, md_p = write_outputs(
        "few_shot_supplementary",
        rows,
        md_table=md_table,
        script_name="paper_artifacts.appendix.few_shot_supplementary",
        md_caption=(
            "**Few-Shot Supplementary Check.** Few-shot supplementary check (single-seed "
            "GPT-5.4, k=3 exemplars per question). Forced accuracy, selective "
            "accuracy, skip rate, and per-type accuracy on F-Missing-Data and "
            "G-Annotation. DSNBF baseline reproduced by `per_type_accuracy.py`."
        ),
        md_footnotes=[
            "*Paper location: `Appendix selective-QA section` "
            "(lines 43-50) and `s4_experiments.tex` (few-shot narrative).*",
            f"*Tolerance: ±{PAPER_TOLERANCE} absolute on every fraction (≈ ±0.5 pp).*",
            f"*Few-shot frozen artifact: 120 personas × 18 qids "
            f"= 2160 calls (`{_FS_DIR.relative_to(_REPO_ROOT)}`).*",
        ],
        subdir="appendix",
    )
    print(f"  ↳ wrote {md_p.relative_to(_REPO_ROOT.parent)}")
    print(f"  ↳ wrote {csv_p.relative_to(_REPO_ROOT.parent)}")

    n_fail = sum(1 for r in rows if isinstance(r["paper_match"], str)
                 and r["paper_match"].startswith("FAIL"))
    return n_fail


def main() -> int:
    return run()


if __name__ == "__main__":
    sys.exit(main())
