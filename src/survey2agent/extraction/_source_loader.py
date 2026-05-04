"""Source loader for oracle ExtractedAtom builder.

Lifted verbatim from v1.0 reference:22-37. The only deviation is the explicit
``encoding="utf-8"`` argument on each ``read_text`` call (Windows safety);
the JSON parse output is byte-identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_sources(persona_dir: Path) -> dict[str, Any]:
    """Load the five raw source JSONs for a persona.

    The returned dict has the exact shape ``compute_all_mu`` expects:
    ``profile_ltm`` is the full nested object; the other four are unwrapped
    to their ``records`` list.
    """
    src_dir = persona_dir / "structural_sources"
    profile = json.loads((src_dir / "profile_ltm.json").read_text(encoding="utf-8"))
    planner = json.loads((src_dir / "planner.json").read_text(encoding="utf-8"))
    self_report = json.loads(
        (src_dir / "daily_self_report.json").read_text(encoding="utf-8")
    )
    objective = json.loads(
        (src_dir / "objective_log.json").read_text(encoding="utf-8")
    )
    device = json.loads((src_dir / "device_log.json").read_text(encoding="utf-8"))
    return {
        "profile_ltm": profile,
        "planner": planner.get("records", []),
        "daily_self_report": self_report.get("records", []),
        "objective_log": objective.get("records", []),
        "device_log": device.get("records", []),
    }
