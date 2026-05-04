"""CLI entry point for L4 — Ground truth computation.

Reads event_table.json and structural_sources/ per persona, then writes
ground_truth.json with deterministic answers for all 18 benchmark questions.

Usage
-----
    python -m survey2agent.data_generation.generate_ground_truth --dataset-dir <dir>
    python -m survey2agent.data_generation.generate_ground_truth --dataset-dir <dir> --persona <id>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .ground_truth import compute_all_ground_truths


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_source_records(source_path: Path) -> list[dict[str, Any]]:
    """Load a source JSON and return its records list."""
    if not source_path.exists():
        return []
    data = _load_json(source_path)
    return data.get("records", [])


def process_persona(
    persona: dict[str, Any],
    persona_dir: Path,
) -> dict[str, dict[str, Any]]:
    """Compute ground truth for a single persona."""
    event_table = _load_json(persona_dir / "event_table.json")

    sources_dir = persona_dir / "structural_sources"
    profile_ltm_path = sources_dir / "profile_ltm.json"
    sources = {
        "planner": _load_source_records(sources_dir / "planner.json"),
        "device_log": _load_source_records(sources_dir / "device_log.json"),
        "objective_log": _load_source_records(sources_dir / "objective_log.json"),
        "profile_ltm": _load_json(profile_ltm_path) if profile_ltm_path.exists() else {},
    }

    return compute_all_ground_truths(event_table, persona, sources)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute ground truth answers for 18 benchmark questions (L4)"
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Root directory of the generated dataset",
    )
    parser.add_argument(
        "--persona",
        default=None,
        help="Process a single persona by ID (default: all)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    personas_path = dataset_dir / "config" / "personas.json"
    if not personas_path.exists():
        print(f"ERROR: {personas_path} not found", file=sys.stderr)
        return 1

    personas_data = _load_json(personas_path)
    personas = personas_data["personas"]

    if args.persona:
        personas = [p for p in personas if p["id"] == args.persona]
        if not personas:
            print(f"ERROR: persona '{args.persona}' not found", file=sys.stderr)
            return 1

    t0 = time.perf_counter()
    count = 0

    for persona in personas:
        pid = persona["id"]
        persona_dir = dataset_dir / pid

        if not (persona_dir / "event_table.json").exists():
            print(f"SKIP {pid}: no event_table.json")
            continue

        gt = process_persona(persona, persona_dir)
        _save_json(persona_dir / "ground_truth.json", gt)
        count += 1

    elapsed = time.perf_counter() - t0
    print(
        f"Done: {count} personas, 18 questions each = {count * 18} GT labels "
        f"in {elapsed:.1f}s ({elapsed / max(1, count) * 1000:.0f}ms/persona)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
