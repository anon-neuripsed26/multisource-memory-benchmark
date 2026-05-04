"""CLI entry point for L3 source projection.

Usage::

    python -m survey2agent.data_generation.generate_sources \
        --dataset-dir data/benchmark/seeds/s20260321

Reads ``<dataset-dir>/config/personas.json`` and each persona's
``event_table.json``, then writes 5 source JSONs + metadata into
``<dataset-dir>/<persona_id>/structural_sources/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .source_projector import project_all_sources


SOURCE_DIR = "structural_sources"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate L3 structural source projections for all personas",
    )
    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="Benchmark dataset root (must contain config/personas.json "
             "and per-persona event_table.json files)",
    )
    parser.add_argument(
        "--persona", type=str, default=None,
        help="Generate for a single persona ID only",
    )
    parser.add_argument(
        "--knob-scale-bias", type=float, default=1.0,
        help="Multiplicative scale for bias knobs (default: 1.0)",
    )
    parser.add_argument(
        "--knob-scale-dropout", type=float, default=1.0,
        help="Multiplicative scale for dropout knobs (default: 1.0)",
    )
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir).resolve()
    personas_path = dataset_dir / "config" / "personas.json"

    if not personas_path.exists():
        print(f"ERROR: {personas_path} not found", file=sys.stderr)
        return 1

    data = _load_json(personas_path)
    personas: list[dict[str, Any]] = data["personas"]

    # Read dataset seed from personas.json metadata
    pilot_config = data.get("pilot_config", {})
    base_seed = pilot_config.get("random_seed", 0)
    if base_seed == 0:
        base_seed = data.get("benchmark_metadata", {}).get("seed", 20260321)

    if args.persona:
        personas = [p for p in personas if p["id"] == args.persona]
        if not personas:
            print(f"ERROR: persona '{args.persona}' not found", file=sys.stderr)
            return 1

    print(f"Using base_seed={base_seed} for source projection RNG")

    # Build knob overrides from scale factors
    BIAS_KNOBS = [
        "self_report_conflict_rate", "self_report_underreport_bias",
        "self_report_overreport_bias", "planner_optimism_bias",
        "planner_behavior_gap_rate",
    ]
    DROPOUT_KNOBS = [
        "device_dropout_rate", "device_noise_rate",
        "objective_dropout_rate", "objective_noise_rate",
    ]
    knob_overrides: dict[str, Any] | None = None
    if args.knob_scale_bias != 1.0 or args.knob_scale_dropout != 1.0:
        print(f"  Knob scaling: bias={args.knob_scale_bias}x, dropout={args.knob_scale_dropout}x")
        # Overrides are applied per-persona inside the loop (computed from defaults)
        knob_scale = {"bias": args.knob_scale_bias, "dropout": args.knob_scale_dropout}
    else:
        knob_scale = None

    t0 = time.perf_counter()
    generated = 0

    for persona in personas:
        pid = persona["id"]
        event_path = dataset_dir / pid / "event_table.json"

        if not event_path.exists():
            print(f"  SKIP {pid} — no event_table.json", file=sys.stderr)
            continue

        records = _load_json(event_path)

        # Compute per-persona knob overrides from scale factors
        per_persona_overrides = None
        if knob_scale:
            from .source_projector import _infer_knobs
            defaults = _infer_knobs(persona, len(records))
            per_persona_overrides = {}
            for k in BIAS_KNOBS:
                if k in defaults:
                    per_persona_overrides[k] = min(1.0, defaults[k] * knob_scale["bias"])
            for k in DROPOUT_KNOBS:
                if k in defaults:
                    per_persona_overrides[k] = min(0.50, defaults[k] * knob_scale["dropout"])

        sources = project_all_sources(persona, records, base_seed=base_seed,
                                      knob_overrides=per_persona_overrides)

        out_dir = dataset_dir / pid / SOURCE_DIR
        for source_name, payload in sources.items():
            _save_json(out_dir / f"{source_name}.json", payload)

        generated += 1
        if generated <= 3 or generated % 50 == 0 or generated == len(personas):
            elapsed = time.perf_counter() - t0
            print(
                f"  [{generated}/{len(personas)}] {pid} "
                f"-> {out_dir.relative_to(dataset_dir)} ({elapsed:.1f}s)"
            )

    elapsed = time.perf_counter() - t0
    print(
        f"\nGenerated sources for {generated} personas in {elapsed:.1f}s "
        f"({elapsed / max(generated, 1) * 1000:.0f}ms/persona)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
