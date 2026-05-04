"""Generate a valid sample leaderboard submission for the `Random` baseline.

Walks `data/benchmark/seeds/<seed>/` to discover real persona names, runs
`survey2agent.methods.Random` over all 18 questions for each persona, and
writes one schema-valid JSON file per `(seed, persona)` to
`<output_dir>/<seed>/<persona>.json`.

The output is what an external contributor's submission must look like.
After running, validate with:

    python3 -m paper_artifacts.verify_external_submission \\
        --predictions examples/04_sample_submission/out \\
        --method-name SampleRandom-2026 --strict

Usage:

    python3 examples/04_sample_submission/make_submission.py \\
        --output-dir examples/04_sample_submission/out \\
        --seed s20260321
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS
from survey2agent.methods import Random as RandomMethod

REPO_ROOT = Path(__file__).resolve().parents[2]
METHOD_NAME = "SampleRandom-2026"


def iter_persona_names(seed_dir: Path) -> list[str]:
    return sorted(p.name for p in seed_dir.iterdir() if p.is_dir())


def build_submission_payload(method: RandomMethod, seed: str, persona: str) -> dict:
    atom = ExtractedAtom(
        persona=persona,
        extraction={qid: {} for qid in QUESTIONS},
    )
    predictions: dict[str, dict] = {}
    for qid in QUESTIONS:
        pred = method.predict_one(atom, qid)
        predictions[qid] = {
            "answer": pred.answer,
            "would_skip": pred.would_skip,
        }
        if pred.raw_answer is not None:
            predictions[qid]["raw_answer"] = pred.raw_answer
    return {
        "method_name": METHOD_NAME,
        "seed": seed,
        "persona_id": persona,
        "predictions": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "out",
    )
    parser.add_argument("--seed", default="s20260321")
    args = parser.parse_args()

    seed_dir = REPO_ROOT / "data" / "benchmark" / "seeds" / args.seed
    if not seed_dir.exists():
        sys.exit(
            f"Seed directory not found: {seed_dir}\n"
            "Run `make fetch` first."
        )

    out_seed_dir = args.output_dir / args.seed
    out_seed_dir.mkdir(parents=True, exist_ok=True)

    method = RandomMethod(seed=42)

    n = 0
    for persona in iter_persona_names(seed_dir):
        payload = build_submission_payload(method, args.seed, persona)
        (out_seed_dir / f"{persona}.json").write_text(json.dumps(payload, indent=2))
        n += 1

    print(f"Wrote {n} submission files to {out_seed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
