"""Reproduce the API-call-inventory table — cache-file count cross-check.

Source: ``Appendix method-details section`` §Compute
Footprint, ``\\label{tab:api_inventory}`` ("API call inventory and
measured token summary").

Per the internal cache-inventory audit (§ 5e2), this lock
is *defense in depth* against accidental cache deletion rather than a
measured-outcome check — the paper's printed counts are design totals
(4 seeds x 480 personas for extraction, etc.), while the released
``data/method_outputs/`` cache retains a test-split subset
for extraction rows.

For each of the 8 API-stage rows we record:

  * ``paper_calls``         — API call count printed in the paper cell
    (design total incurred at the time of the original run).
  * ``expected_cache_files`` — number of JSON files expected in the
    released ``data/`` cache backing this row.
  * ``cache_root`` + glob    — the path(s) enumerated to produce the
    measured file count.
  * ``scope_note``           — one-line note explaining any gap
    between ``paper_calls`` and ``expected_cache_files``.

Lock criterion: ``measured_file_count == expected_cache_files``
(exact integer match; no tolerance). Drift detected here indicates the
released cache has been truncated or mis-routed.

Two rows have ``expected_cache_files < paper_calls``:

  * ``gpt_extraction`` / ``gemini_extraction``: released cache keeps
    only the 120-persona test split per seed (480 files), whereas
    the paper's 1,920 figure counts the original full 480-persona
    extraction runs per seed (training / dev / cal atoms are derived
    from the GT-backed oracle read-out, not the LLM cache).

The other 6 rows match paper counts exactly.

Reproduces against ``data/extracted_atoms/{seed}/`` and
``data/method_outputs/{model}/{seed}/{variant}/`` (plus
``method_outputs/struct_llm/{extracted,oracle}/{seed}/``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .._common import OUTPUT_DIR


# Paper row order (as printed in the .tex table body, top-to-bottom).
PAPER_ROW_ORDER: tuple[str, ...] = (
    "gpt_extraction",
    "gemini_extraction",
    "gpt_direct_schema",
    "gpt_struct_llm",
    "gemini_direct_schema",
    "deepseek_direct_schema",
    "qwen_direct_schema",
    "fewshot_seed1",
)

# Human-readable label per row (matches paper's Stage column phrasing).
ROW_LABEL: dict[str, str] = {
    "gpt_extraction":               "GPT-5.4 extraction (T1/T2 input)",
    "gemini_extraction":            "Gemini 3.1 Pro extraction (cross-extractor robustness)",
    "gpt_direct_schema":            "GPT-5.4 LLM-Direct + Schema-Aware (T3)",
    "gpt_struct_llm":               "GPT-5.4 on atoms (mu_hat + mu_star)",
    "gemini_direct_schema":         "Gemini 3.1 Pro LLM-Direct + Schema-Aware (T3)",
    "deepseek_direct_schema":       "DeepSeek V3.2 LLM-Direct + Schema-Aware (T3)",
    "qwen_direct_schema":           "Qwen3-235B LLM-Direct + Schema-Aware (T3)",
    "fewshot_seed1":                "Few-shot pilot (seed 1, per-question)",
}

# Paper-printed API call count per row ("Calls" column).
PAPER_TAB_API_INVENTORY: dict[str, int] = {
    "gpt_extraction":               1920,
    "gemini_extraction":            1920,
    "gpt_direct_schema":             960,
    "gpt_struct_llm":                960,
    "gemini_direct_schema":          960,
    "deepseek_direct_schema":        960,
    "qwen_direct_schema":            960,
    "fewshot_seed1":                2160,
}

PAPER_TOTAL_CALLS: int = 10800  # paper "Total" cell

# Expected file counts in the released cache for each row.
# For the rows where paper_calls > expected (extraction test-split
# retention), the gap is documented in ROW_SCOPE_NOTE below.
# Lock criterion is measured == expected.
EXPECTED_CACHE_COUNTS: dict[str, int] = {
    "gpt_extraction":                480,   # 120 test x 4 seeds
    "gemini_extraction":             480,   # 120 test x 4 seeds
    "gpt_direct_schema":             960,   # 120 x 4 x 2 (direct + schema-aware)
    "gpt_struct_llm":                960,   # 120 x 4 x 2 (mu_hat + mu_star atoms)
    "gemini_direct_schema":          960,   # 120 x 4 x 2
    "deepseek_direct_schema":        960,   # 120 x 4 x 2 (direct + schema-aware)
    "qwen_direct_schema":            960,   # 120 x 4 x 2 (direct + schema-aware)
    "fewshot_seed1":                2160,   # 120 x 18 questions, seed 1 only
}

ROW_SCOPE_NOTE: dict[str, str] = {
    "gpt_extraction":
        "released cache retains test-split atoms (120 x 4 = 480); "
        "paper 1,920 = full 480-persona extraction across 4 seeds",
    "gemini_extraction":
        "released cache retains test-split atoms (120 x 4 = 480); "
        "paper 1,920 = full 480-persona extraction across 4 seeds",
    "gpt_direct_schema":            "full release coverage",
    "gpt_struct_llm":               "full release coverage (mu_hat + mu_star atoms)",
    "gemini_direct_schema":         "full release coverage",
    "deepseek_direct_schema":       "full release coverage",
    "qwen_direct_schema":           "full release coverage",
    "fewshot_seed1":                "full release coverage (seed-1 pilot)",
}


# ── Cache-path descriptors ───────────────────────────────────────────────

# Each row maps to a list of (root, glob) pairs; measured count is the
# sum of ``len(list(root.glob(glob)))`` across pairs.
_DATA_ROOT: Path = Path(__file__).resolve().parents[2] / "data"
_EXTRACTED_ATOMS: Path = _DATA_ROOT / "extracted_atoms"
_METHOD_OUTPUTS: Path = _DATA_ROOT / "method_outputs"

_SEEDS: tuple[str, ...] = ("s20260321", "s20260322", "s20260323", "s20260324")


def _row_cache_paths(row: str) -> list[tuple[Path, str]]:
    """Return (root, glob) pairs whose matching JSON files back ``row``."""
    if row == "gpt_extraction":
        return [(_EXTRACTED_ATOMS / s, "*.json") for s in _SEEDS]
    if row == "gemini_extraction":
        return [(_METHOD_OUTPUTS / "gemini_p2" / s / "extract", "*.json") for s in _SEEDS]
    if row == "gpt_direct_schema":
        pairs: list[tuple[Path, str]] = []
        for s in _SEEDS:
            pairs.append((_METHOD_OUTPUTS / "gpt-5.4" / s / "direct", "*.json"))
            pairs.append((_METHOD_OUTPUTS / "gpt-5.4" / s / "schema-aware", "*.json"))
        return pairs
    if row == "gpt_struct_llm":
        pairs = []
        for arm in ("extracted", "oracle"):
            for s in _SEEDS:
                pairs.append((_METHOD_OUTPUTS / "struct_llm" / arm / s, "*.json"))
        return pairs
    if row == "gemini_direct_schema":
        pairs = []
        for s in _SEEDS:
            pairs.append((_METHOD_OUTPUTS / "gemini_p2" / s / "direct", "*.json"))
            pairs.append((_METHOD_OUTPUTS / "gemini_p2" / s / "schema-aware", "*.json"))
        return pairs
    if row == "deepseek_direct_schema":
        pairs = []
        for s in _SEEDS:
            pairs.append((_METHOD_OUTPUTS / "deepseek-v3.2" / s / "direct", "*.json"))
            pairs.append((_METHOD_OUTPUTS / "deepseek-v3.2" / s / "schema-aware", "*.json"))
        return pairs
    if row == "qwen_direct_schema":
        pairs = []
        for s in _SEEDS:
            pairs.append((_METHOD_OUTPUTS / "qwen3-235b-a22b-2507" / s / "direct", "*.json"))
            pairs.append((_METHOD_OUTPUTS / "qwen3-235b-a22b-2507" / s / "schema-aware", "*.json"))
        return pairs
    if row == "fewshot_seed1":
        return [(_METHOD_OUTPUTS / "gpt-5.4" / "s20260321" / "few-shot", "*.json")]
    raise ValueError(f"unknown row {row!r}")


# ── Computation ──────────────────────────────────────────────────────────


def compute_inventory() -> dict[str, int]:
    """Count JSON files backing each paper-table row in the released cache."""
    counts: dict[str, int] = {}
    for row in PAPER_ROW_ORDER:
        total = 0
        for root, pattern in _row_cache_paths(row):
            if root.exists():
                total += len(list(root.glob(pattern)))
        counts[row] = total
    return counts


# ── Checker ──────────────────────────────────────────────────────────────


def check_cells(
    measured: dict[str, int],
) -> tuple[int, int, list[str], dict[str, int]]:
    """Compare measured file counts to ``EXPECTED_CACHE_COUNTS``.

    Returns ``(n_pass, n_total, fails, measured)``. ``n_total`` covers
    the 8 row cells plus the aggregate Total cell (9 cells).
    """
    fails: list[str] = []
    n_pass = 0
    n_total = 0
    for row in PAPER_ROW_ORDER:
        exp = EXPECTED_CACHE_COUNTS[row]
        meas = measured[row]
        n_total += 1
        if meas == exp:
            n_pass += 1
        else:
            fails.append(
                f"row {row}: measured={meas} expected={exp} "
                f"(paper prints {PAPER_TAB_API_INVENTORY[row]}; "
                f"note: {ROW_SCOPE_NOTE[row]})"
            )
    # Total-cell check (sum of expected cache counts, not paper total).
    expected_total = sum(EXPECTED_CACHE_COUNTS.values())
    measured_total = sum(measured.values())
    n_total += 1
    if measured_total == expected_total:
        n_pass += 1
    else:
        fails.append(
            f"total: measured={measured_total} expected={expected_total} "
            f"(paper prints {PAPER_TOTAL_CALLS})"
        )
    return n_pass, n_total, fails, measured


# ── Output ───────────────────────────────────────────────────────────────


def _render_outputs(
    measured: dict[str, int],
    n_pass: int,
    n_total: int,
    fails: list[str],
) -> tuple[Path, Path]:
    out_dir = OUTPUT_DIR / "appendix"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "api_cache_inventory.csv"
    md_path = out_dir / "api_cache_inventory.md"

    csv_lines: list[str] = [
        "# Generated by paper_artifacts.appendix.api_cache_inventory",
        "# Tolerance: exact integer match on cache file count",
        "row,label,paper_calls,expected_cache_files,measured_cache_files,paper_match,scope_note",
    ]
    for row in PAPER_ROW_ORDER:
        exp = EXPECTED_CACHE_COUNTS[row]
        meas = measured[row]
        paper = PAPER_TAB_API_INVENTORY[row]
        ok = (meas == exp)
        # CSV-safe scope note (strip commas).
        note = ROW_SCOPE_NOTE[row].replace(",", ";")
        csv_lines.append(
            f"{row},{ROW_LABEL[row]},{paper},{exp},{meas},"
            f"{'OK' if ok else 'FAIL'},{note}"
        )
    # Total line
    exp_total = sum(EXPECTED_CACHE_COUNTS.values())
    meas_total = sum(measured.values())
    ok_total = (meas_total == exp_total)
    csv_lines.append(
        f"TOTAL,(sum),{PAPER_TOTAL_CALLS},{exp_total},{meas_total},"
        f"{'OK' if ok_total else 'FAIL'},"
        f"paper 10;800 = total API calls; released cache retains "
        f"{exp_total} files (test-split extraction only)"
    )
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    md_lines: list[str] = [
        "# api_cache_inventory",
        "",
        "**API call inventory.** Cache-file count cross-check "
        "(defense-in-depth lock against accidental cache deletion). Each row "
        "lists the paper's printed API call count, the expected number of "
        "JSON files in the released `data/` cache backing that "
        "row, and the measured count. Lock criterion: measured == expected "
        "(exact integer; drift indicates cache truncation).",
        "",
        "| Row | Paper calls | Expected cache | Measured | Status | Scope |",
        "|:---|---:|---:|---:|:---:|:---|",
    ]
    for row in PAPER_ROW_ORDER:
        exp = EXPECTED_CACHE_COUNTS[row]
        meas = measured[row]
        paper = PAPER_TAB_API_INVENTORY[row]
        ok = (meas == exp)
        md_lines.append(
            f"| `{row}` | {paper} | {exp} | {meas} | "
            f"{'OK' if ok else 'FAIL'} | {ROW_SCOPE_NOTE[row]} |"
        )
    md_lines.append(
        f"| **Total** | **{PAPER_TOTAL_CALLS}** | **{exp_total}** | "
        f"**{meas_total}** | **{'OK' if ok_total else 'FAIL'}** | "
        f"paper 10,800 = total calls; released cache = test-split "
        f"extraction + full T3/Struct/Few-shot |"
    )
    md_lines += [
        "",
        f"*Cells locked: {n_pass}/{n_total} (exact integer match on "
        f"released cache file counts).*",
        "",
        "**Scope notes.** Two rows have expected cache < paper calls:",
        "",
        "- `gpt_extraction` / `gemini_extraction`: released cache keeps "
        "the 120-persona test split (480 files each); the paper's 1,920 "
        "figure counts the original full 480-persona extraction runs "
        "across 4 seeds. Training / dev / cal atoms for those rows are "
        "reconstructed deterministically from the GT-backed oracle "
        "read-out rather than re-cached.",
        "",
        "*Source: `data/extracted_atoms/{seed}/` and "
        "`data/method_outputs/{model,struct_llm}/...`.*",
        "",
        "*Paper location: `Appendix method-details section`, "
        "§Compute Footprint, `\\label{tab:api_inventory}` "
        "(API call inventory and measured token summary).*",
        "",
        "*Released-cache total `expected = 7920 = 480 + 480 + 960 + 960 + "
        "960 + 960 + 960 + 2160`; paper-printed `total = 10,800` includes "
        "the original 1,920 + 1,920 extraction calls before test-split "
        "trimming.*",
    ]
    if fails:
        md_lines.append("")
        md_lines.append("### Failures")
        for msg in fails:
            md_lines.append(f"- {msg}")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return csv_path, md_path


# ── Entry point ──────────────────────────────────────────────────────────


def main() -> int:
    t0 = time.time()
    print("[api_cache_inventory] enumerating released cache files for "
          "API-inventory rows...", flush=True)
    measured = compute_inventory()
    n_pass, n_total, fails, _ = check_cells(measured)

    csv_path, md_path = _render_outputs(measured, n_pass, n_total, fails)

    elapsed = time.time() - t0
    print(f"\n=== API Cache Inventory ===")
    print(f"Cells: {n_pass}/{n_total} "
          f"({n_pass * 100 / n_total:.1f}%) in {elapsed:.1f}s")
    print(f"Outputs: {csv_path.name}, {md_path.name}")
    for msg in fails:
        print(f"  FAIL: {msg}")
    return 0 if n_pass == n_total else (n_total - n_pass)


if __name__ == "__main__":
    sys.exit(main())
