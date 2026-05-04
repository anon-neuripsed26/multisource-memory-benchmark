"""CLI entry point for L2 event table generation.

Usage::

    python -m survey2agent.data_generation.generate_events \
        --dataset-dir data/benchmark/seeds/s20260321

Reads ``<dataset-dir>/config/personas.json`` and writes per-persona event
tables into ``<dataset-dir>/<persona_id>/event_table.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .event_generator import generate_event_table


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate L2 event tables for all personas in a dataset",
    )
    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="Benchmark dataset root (must contain config/personas.json)",
    )
    parser.add_argument(
        "--persona", type=str, default=None,
        help="Generate for a single persona ID only",
    )
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir).resolve()
    personas_path = dataset_dir / "config" / "personas.json"

    if not personas_path.exists():
        print(f"ERROR: {personas_path} not found", file=sys.stderr)
        return 1

    with open(personas_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    personas = data["personas"]
    pilot_config = data["pilot_config"]
    start_date = pilot_config["start_date"]
    num_days = pilot_config["num_days"]
    seed = pilot_config.get("random_seed", 0)

    # Infer seed from benchmark_metadata if not in pilot_config
    if seed == 0:
        seed = data.get("benchmark_metadata", {}).get("seed", 20260321)

    total = len(personas)
    if args.persona:
        personas = [p for p in personas if p["id"] == args.persona]
        if not personas:
            print(f"ERROR: persona '{args.persona}' not found", file=sys.stderr)
            return 1

    t0 = time.perf_counter()
    generated = 0

    for i, persona in enumerate(personas, 1):
        pid = persona["id"]
        events = generate_event_table(
            persona,
            start_date=start_date,
            num_days=num_days,
            base_seed=seed,
        )

        out_path = dataset_dir / pid / "event_table.json"
        _save_json(out_path, events)
        generated += 1

        if generated <= 3 or generated % 50 == 0 or generated == len(personas):
            elapsed = time.perf_counter() - t0
            print(f"  [{generated}/{len(personas)}] {pid} -> {len(events)} days ({elapsed:.1f}s)")

    elapsed = time.perf_counter() - t0
    print(f"\nGenerated {generated} event tables in {elapsed:.1f}s "
          f"({elapsed / max(generated, 1) * 1000:.0f}ms/persona)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
