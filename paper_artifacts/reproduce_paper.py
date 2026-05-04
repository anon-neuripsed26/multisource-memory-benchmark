"""CLI orchestrator for paper-table reproduction.

Two-tier layout: `main/` reproduces the four main paper tables; `appendix/`
reproduces the nineteen appendix tables (per-type / per-difficulty
breakdowns, robustness grids, prediction distributions, supplementary
checks, and the API-call inventory).

Usage::

    python -m paper_artifacts.reproduce_paper                          # all tiers
    python -m paper_artifacts.reproduce_paper --tier main
    python -m paper_artifacts.reproduce_paper --tier appendix
    python -m paper_artifacts.reproduce_paper --names forced_accuracy,selective_qa

Exit code is the total number of cells outside ±0.005 tolerance across
all run scripts (0 = within tolerance, 1+ = drift detected).

Wall-clock guidance (frozen artifacts, no LLM API calls):

  * full `make reproduce`      : ~1-2 hours on an Apple M1 Pro laptop
                                  with 16 GB RAM (CPU-only)
  * `selective_qa_full`         : ~1-2 minutes (per-seed mean, no bootstrap)
  * `factorial_decomposition`   : ~3-5 minutes (4 cells x 2000-resample CI)
  * `forced_accuracy_main`      : ~10-15 minutes (17 rows x 2000-resample CI)
  * `per_type_accuracy`         : ~12-18 minutes (8 LLM x per-type + paired delta)
  * each appendix script        : <30 s to ~3 min (see paper_artifacts/MANIFEST.md)

Prerequisites:
  * `data/method_outputs/<model>/<seed>/<variant>/` populated
    with the four LLM frozen artifact families (and `struct_llm/`).
  * `$S2A_DATA_ROOT/benchmark/seeds/<seed>/` available for oracle atoms.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path


# Registry: tier -> {script_name: human_label}
# Each label is "<descriptive name> (<paper subsection title>)" so reader can
# locate the corresponding section in the paper without depending on the
# appendix letter / table number, which can shift across revisions.
_REGISTRY: dict[str, dict[str, str]] = {
    "main": {
        "selective_qa_full":       "selective QA full table (Full Selective QA Table)",
        "factorial_decomposition": "2x2 factorial decomposition (Factorial Decomposition: Resolver x Input)",
        "forced_accuracy_main":    "answer-only accuracy + bootstrap CI (Main Results and Selective QA)",
        "per_type_accuracy":       "per-type diagnostic (Diagnostic Analysis by Reasoning Type)",
    },
    "appendix": {
        "per_type_macro_accuracy_full":              "full per-type macro accuracy (Per-Type Accuracy)",
        "t2_fusion_per_type_per_difficulty":         "T2 fusion per-type x per-diff (Difficulty-Class Breakdown, T2 fusion)",
        "t3_llm_per_type_per_difficulty":            "T3 LLM   per-type x per-diff (Difficulty-Class Breakdown, T3 LLM)",
        "t2_fusion_selective_per_type_per_difficulty": "T2 +SKIP per-type x per-diff (Difficulty-Class Breakdown, T2 +SKIP)",
        "t3_llm_selective_per_type_per_difficulty":    "T3 +SKIP per-type x per-diff (Difficulty-Class Breakdown, T3 +SKIP)",
        "prediction_distributions_e_causal":          "E1/E2 prediction histograms (Prediction Distributions on Failure Questions, E-type)",
        "prediction_distributions_c_pr_f_miss":       "C2/F3 prediction histograms (Prediction Distributions on Failure Questions, C/F-type)",
        "cross_seed_stability":                       "cross-seed stability (Cross-Seed Stability)",
        "train_size_ablation":                        "training-size ablation (Training Size Sensitivity)",
        "cross_condition_gpt_vs_gemini":              "GPT vs Gemini cross-condition, seed 1 (GPT-5.4 vs Gemini Cross-Condition Comparison)",
        "dgp_perturbation":                           "DGP perturbation grid 9x7 (DGP Perturbation)",
        "noise_perturbation":                         "extraction noise tolerance (Extraction Noise Tolerance, Full Analysis)",
        "cross_extractor_robustness":                 "cross-extractor robustness (Cross-Extractor Robustness)",
        "cross_bias_transfer":                        "cross-parameter transfer without refit, DSNBF (Cross-Parameter Transfer Without Refit; ~9 min)",
        "per_question_extraction_accuracy":           "per-Q x per-diff extraction accuracy (Cross-Extractor Robustness, per-question subtable)",
        "atom_extraction_faithfulness":               "atom extraction faithfulness audit (Atom Extraction Faithfulness Audit; inline prose)",
        "source_ceiling_complement_table":            "source-reachability complement diagnostic (GT Construction Notes, complement table)",
        "api_cache_inventory":                        "API call inventory / cache-file cross-check (Compute Footprint)",
        "few_shot_supplementary":                     "single-seed few-shot supplementary check (Few-Shot Supplementary Check)",
    },
}


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _print_banner(selected: list[tuple[str, str]], out_dir: Path) -> None:
    print("=" * 70)
    print("Survey2Agent — Paper Table Reproduction")
    print("=" * 70)
    print(f"Scripts to run    : {len(selected)}")
    for tier, name in selected:
        print(f"  - {tier}.{name}")
    print(f"Output directory  : {out_dir}")
    print("Prerequisites: data/method_outputs/{<model>,struct_llm} "
          "and $S2A_DATA_ROOT/benchmark/seeds/<seed>/")
    print("Runtime guide      : full run is about 1-2h on an Apple M1 Pro "
          "laptop with 16 GB RAM; progress is printed before and after "
          "each script.")
    print("=" * 70)


def _resolve(tier: str | None, names: list[str] | None) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    tiers = [tier] if tier and tier != "all" else list(_REGISTRY.keys())
    name_filter = set(names) if names else None
    for t in tiers:
        if t not in _REGISTRY:
            raise ValueError(f"unknown tier {t!r}; valid: {sorted(_REGISTRY)}")
        for n in _REGISTRY[t]:
            if name_filter is None or n in name_filter:
                selected.append((t, n))
    return selected


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "output"


def _data_root() -> Path:
    override = os.environ.get("S2A_DATA_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "data"


def _preflight_data_layout() -> None:
    """Fail early with a clear fetch hint instead of cascading stack traces."""
    root = _data_root()
    required = (
        root / "benchmark" / "seeds",
        root / "benchmark" / "results",
        root / "extracted_atoms",
        root / "method_outputs",
    )
    missing = [path for path in required if not path.exists()]
    if not missing:
        return

    rels = "\n".join(f"  - {path}" for path in missing)
    raise SystemExit(
        "Missing full benchmark data required for paper reproduction.\n"
        f"Data root: {root}\n"
        f"Missing paths:\n{rels}\n\n"
        "Run `make fetch` first, or set S2A_DATA_ROOT to a fetched data "
        "directory containing benchmark/, extracted_atoms/, and method_outputs/."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier", default="all", choices=["main", "appendix", "all"],
        help="which tier to run (default: all)",
    )
    parser.add_argument(
        "--names", default="",
        help="comma-separated script names within the chosen tier "
             "(default: all scripts in tier)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="override output root (default: paper_artifacts/output/)",
    )
    args = parser.parse_args()

    name_filter = [n.strip() for n in args.names.split(",") if n.strip()] or None
    try:
        selected = _resolve(args.tier, name_filter)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not selected:
        print(f"no scripts matched (tier={args.tier!r}, names={name_filter!r})",
              file=sys.stderr)
        return 2

    output_dir = Path(args.out_dir) if args.out_dir else _default_output_dir()

    _preflight_data_layout()
    _print_banner(selected, output_dir)

    if args.out_dir:
        from . import _common
        _common.OUTPUT_DIR = output_dir

    total_fail = 0
    t_overall = time.time()
    total = len(selected)
    durations: list[float] = []
    for idx, (tier, name) in enumerate(selected, start=1):
        label = _REGISTRY[tier][name]
        elapsed = time.time() - t_overall
        if durations:
            avg = sum(durations) / len(durations)
            remaining = avg * (total - idx + 1)
            eta = f"; ETA ~{_fmt_duration(remaining)}"
        else:
            eta = ""
        print(
            f"\n>>> [{idx}/{total}] Running {tier}.{name} — {label}\n"
            f"    elapsed {_fmt_duration(elapsed)}{eta}",
            flush=True,
        )
        t0 = time.time()
        try:
            mod = importlib.import_module(f"paper_artifacts.{tier}.{name}")
            fn = getattr(mod, "main")
            fail = fn()
        except Exception as exc:
            dt = time.time() - t0
            durations.append(dt)
            print(
                f"!!! [{idx}/{total}] {tier}.{name} crashed after "
                f"{_fmt_duration(dt)}: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
            import traceback
            traceback.print_exc()
            total_fail += 1
            continue
        dt = time.time() - t0
        durations.append(dt)
        total_fail += int(fail or 0)
        status = "OK" if (fail or 0) == 0 else f"FAIL ({fail} cells outside tolerance)"
        elapsed = time.time() - t_overall
        if idx < total:
            avg = sum(durations) / len(durations)
            remaining = avg * (total - idx)
            eta = f"; estimated remaining {_fmt_duration(remaining)}"
        else:
            eta = ""
        print(
            f"<<< [{idx}/{total}] {tier}.{name} done in {_fmt_duration(dt)} "
            f"— {status}; elapsed {_fmt_duration(elapsed)}{eta}",
            flush=True,
        )

    overall = time.time() - t_overall
    print("\n" + "=" * 70)
    print(f"Total wall clock: {_fmt_duration(overall)}")
    print(f"Total failed cells: {total_fail}")
    print("=" * 70)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
