"""Programmatic API example.

Show how to compute per-question accuracy of a method without going
through the CLI runner. Useful for ad-hoc experimentation, notebooks,
and unit tests of new methods.

Run:

    python3 examples/03_programmatic_api/compute_accuracy.py

Outputs a table of (qid, n, accuracy) for the AlwaysFirst baseline on
seed s20260321 train split. Wall-clock: ~5 seconds.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS

# Load the AlwaysFirst example without making `examples/` a package.
_REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "always_first",
    _REPO / "examples" / "01_minimal_method" / "always_first.py",
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["always_first"] = _mod
_spec.loader.exec_module(_mod)
AlwaysFirst = _mod.AlwaysFirst


def iter_persona_dirs(seed_dir: Path):
    for p in sorted(seed_dir.iterdir()):
        if p.is_dir() and (p / "ground_truth.json").exists():
            yield p


def main() -> None:
    seed_dir = _REPO / "data" / "benchmark" / "seeds" / "s20260321"
    if not seed_dir.exists():
        sys.exit(
            f"Seed dir not found: {seed_dir}\n"
            "Run `make fetch` first (downloads the benchmark from Hugging Face)."
        )

    method = AlwaysFirst()

    n_correct: dict[str, int] = defaultdict(int)
    n_total: dict[str, int] = defaultdict(int)

    # The minimal method ignores `atom`, so an atom with empty per-qid
    # slots is sufficient for the demo. Real methods would load extracted
    # atoms from each persona directory.
    empty_atom = ExtractedAtom(
        persona="demo",
        extraction={qid: {} for qid in QUESTIONS},
    )

    for persona_dir in iter_persona_dirs(seed_dir):
        gt = json.loads((persona_dir / "ground_truth.json").read_text())
        for qid in QUESTIONS:
            if qid not in gt:
                continue
            pred = method.predict_one(empty_atom, qid)
            n_total[qid] += 1
            if pred.answer == gt[qid].get("answer"):
                n_correct[qid] += 1

    print(f"{'qid':<6} {'n':>6} {'accuracy':>10}")
    for qid in QUESTIONS:
        n = n_total[qid]
        if n == 0:
            continue
        acc = n_correct[qid] / n
        print(f"{qid:<6} {n:>6} {acc:>10.3f}")


if __name__ == "__main__":
    main()
