"""Helper: load per-persona ground truth from `ground_truth.json`.

Schema (per repo memory):
    {"A1": {"question_id": "A1", "answer": "10_to_19", ...}, "A2": {...}, ...}

Returns the flat `{qid: answer_label}` shape that methods consume.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_persona_gt(persona_dir: Path) -> dict[str, str]:
    """Read `persona_dir/ground_truth.json` and return `{qid: answer_label}`.

    Raises:
        FileNotFoundError: if the file is missing.
        ValueError: if a qid entry is malformed (missing `answer` field).
    """
    gt_path = persona_dir / "ground_truth.json"
    with gt_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out: dict[str, str] = {}
    for qid, entry in raw.items():
        if not isinstance(entry, dict) or "answer" not in entry:
            raise ValueError(
                f"GT entry for {qid} in {gt_path} missing 'answer' field"
            )
        out[qid] = entry["answer"]
    return out
