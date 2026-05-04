"""CLI entry point for L1 persona generation.

Usage::

    python -m survey2agent.data_generation.generate_personas \
        --seed 20260321 \
        --output-dir data/benchmark/seeds/s20260321

Produces three files in ``<output-dir>/config/``:
  * ``personas.json``           — 480 persona records
  * ``persona_splits.json``     — stratified split mapping
  * ``benchmark_diversity_audit.json`` — constraint validation report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .constants import (
    END_DATE,
    NUM_DAYS,
    PERSONAS_PER_DIFFICULTY,
    SEMANTIC_CONFLICT_FAMILIES,
    START_DATE,
    SURVEY_REFERENCE_DATE,
    TOTAL_PERSONAS,
)
from .persona_generator import generate_personas
from .split_assigner import SPLIT_RATIOS, assign_splits


# ── I/O helpers ──────────────────────────────────────────────────────────────

def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate V2 benchmark personas (L1: Persona Generation)",
    )
    parser.add_argument(
        "--seed", type=int, required=True,
        help="Deterministic RNG seed (e.g. 20260321)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Benchmark dataset root directory",
    )
    parser.add_argument(
        "--per-difficulty", type=int, default=PERSONAS_PER_DIFFICULTY,
        help=f"Personas per difficulty class (default {PERSONAS_PER_DIFFICULTY})",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate ─────────────────────────────────────────────────────────
    print(f"Generating personas (seed={args.seed}, per_difficulty={args.per_difficulty}) ...")
    personas, audit, attempt = generate_personas(
        seed=args.seed,
        per_difficulty=args.per_difficulty,
    )
    total = len(personas)
    print(f"  Generated {total} personas on attempt {attempt}. Audit passed: {audit['passed']}")

    # ── Splits ───────────────────────────────────────────────────────────
    splits = assign_splits(personas)

    # ── personas.json ────────────────────────────────────────────────────
    dataset_version = output_dir.name
    personas_payload = {
        "personas": personas,
        "pilot_config": {
            "num_days": NUM_DAYS,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "survey_reference_date": SURVEY_REFERENCE_DATE,
        },
        "benchmark_metadata": {
            "version": dataset_version,
            "language": "en",
            "seed": args.seed,
            "generation_attempt": attempt,
            "total_personas": total,
            "per_difficulty": args.per_difficulty,
            "difficulty_types": ["stable", "temporal_shift", "stated_vs_revealed"],
            "split_ratios": {name: ratio for name, ratio in SPLIT_RATIOS},
            "split_summary": splits["summary"],
            "per_track_split_summary": splits["per_track_summary"],
            "diversity_audit_passed": audit["passed"],
            "semantic_conflict_families": list(SEMANTIC_CONFLICT_FAMILIES),
        },
    }
    personas_path = config_dir / "personas.json"
    _save_json(personas_path, personas_payload)

    # ── persona_splits.json ──────────────────────────────────────────────
    splits_path = config_dir / "persona_splits.json"
    _save_json(splits_path, splits)

    # ── benchmark_diversity_audit.json ───────────────────────────────────
    audit_path = config_dir / "benchmark_diversity_audit.json"
    _save_json(audit_path, audit)

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\nOutput files:")
    print(f"  {personas_path}")
    print(f"  {splits_path}")
    print(f"  {audit_path}")
    print(f"\nSplit summary: {splits['summary']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
