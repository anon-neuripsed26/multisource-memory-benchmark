"""Validate an external method's predictions against the leaderboard schema.

Usage:

    python -m paper_artifacts.verify_external_submission \\
        --predictions <dir> \\
        --method-name <alias> \\
        [--seed s20260321] [--strict]

`<dir>` should contain `<seed>/<persona_id>.json` files, one per persona,
each conforming to `schemas/method_prediction.schema.json`.

Exit code:
    0  - all files validate, all required (seed, persona) pairs covered
    1  - schema violation, missing personas, or selective-invariant break

This script is referenced by SUBMISSION_PROTOCOL.md and is the single
gate maintainers run before merging a leaderboard submission.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import jsonschema  # type: ignore
except ImportError:
    sys.exit(
        "jsonschema not installed. Run `pip install jsonschema` "
        "(also pulled in by the package's [dev] extra)."
    )

try:
    import yaml  # type: ignore
except ImportError:
    sys.exit(
        "PyYAML not installed. Run `pip install pyyaml` "
        "(pulled in by the base install)."
    )

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "method_prediction.schema.json"
EXPECTED_QIDS = {
    "A1", "A2", "A3", "B2", "B3", "C2", "C3",
    "D1", "D2", "E1", "E2", "F1", "F2", "F3",
    "G1", "G2", "Ctrl1", "Ctrl2",
}


def _iter_prediction_files(root: Path, seed: str | None) -> list[Path]:
    if seed is not None:
        return sorted((root / seed).glob("*.json"))
    files: list[Path] = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and sub.name.startswith("s"):
            files.extend(sorted(sub.glob("*.json")))
    return files


def _validate_one(
    path: Path,
    schema: dict[str, Any],
    method_name: str,
) -> list[str]:
    """Return a list of error strings (empty if file is fine)."""
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON: {exc}"]

    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError as exc:
        errors.append(f"{path}: schema violation: {exc.message}")
        return errors

    if payload["method_name"] != method_name:
        errors.append(
            f"{path}: method_name {payload['method_name']!r} "
            f"!= expected {method_name!r}"
        )

    if payload["persona_id"] != path.stem:
        errors.append(
            f"{path}: persona_id {payload['persona_id']!r} "
            f"does not match filename stem {path.stem!r}"
        )

    if payload["seed"] != path.parent.name:
        errors.append(
            f"{path}: seed {payload['seed']!r} "
            f"does not match parent dir {path.parent.name!r}"
        )

    qids = set(payload["predictions"].keys())
    if qids != EXPECTED_QIDS:
        missing = EXPECTED_QIDS - qids
        extra = qids - EXPECTED_QIDS
        if missing:
            errors.append(f"{path}: missing predictions for {sorted(missing)}")
        if extra:
            errors.append(f"{path}: unknown qids {sorted(extra)}")

    # Cross-check the SKIP invariant (the schema only enforces shape).
    for qid, pred in payload["predictions"].items():
        ans = pred["answer"]
        ws = pred["would_skip"]
        if (ans == "SKIP") != ws:
            errors.append(
                f"{path}: {qid}: answer/would_skip mismatch "
                f"(answer={ans!r}, would_skip={ws})"
            )
        raw = pred.get("raw_answer")
        if raw == "SKIP":
            errors.append(f"{path}: {qid}: raw_answer must not be 'SKIP'")
        if not ws and raw is not None and raw != ans:
            errors.append(
                f"{path}: {qid}: non-skip raw_answer must equal answer or be null"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--method-name", required=True)
    parser.add_argument(
        "--seed",
        default=None,
        help="Validate only one seed (e.g. s20260321). Default: validate all "
        "seed sub-directories under --predictions.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any seed is missing predictions for any persona that "
        "appears in data/benchmark/seeds/<seed>/.",
    )
    args = parser.parse_args()

    schema = json.loads(SCHEMA_PATH.read_text())

    files = _iter_prediction_files(args.predictions, args.seed)
    if not files:
        sys.exit(f"No prediction files found under {args.predictions}")

    all_errors: list[str] = []
    seeds_seen: set[str] = set()
    personas_per_seed: dict[str, set[str]] = {}

    for path in files:
        seed = path.parent.name
        seeds_seen.add(seed)
        personas_per_seed.setdefault(seed, set()).add(path.stem)
        all_errors.extend(_validate_one(path, schema, args.method_name))

    if args.strict:
        bench_root = REPO_ROOT / "data" / "benchmark" / "seeds"
        if not bench_root.exists():
            sys.exit(
                f"--strict requires the local benchmark to be present at "
                f"{bench_root}. Run `make fetch` first."
            )
        try:
            splits_doc = yaml.safe_load(
                (REPO_ROOT / "configs" / "splits.yaml").read_text()
            )
            test_count = int(
                splits_doc["split"]["partitions"]["test"]["count"]
            )
        except Exception:  # pragma: no cover - splits.yaml is repo-controlled
            test_count = None
        for seed in seeds_seen:
            seed_dir = bench_root / seed
            if not seed_dir.exists():
                all_errors.append(
                    f"--strict: seed {seed}: benchmark directory missing "
                    f"({seed_dir}); run `make fetch`."
                )
                continue
            expected = {p.name for p in seed_dir.iterdir() if p.is_dir()}
            actual = personas_per_seed[seed]
            missing = expected - actual
            if missing:
                all_errors.append(
                    f"--strict: seed {seed}: missing {len(missing)} personas "
                    f"(first 3: {sorted(missing)[:3]})"
                )
            if test_count is not None and len(actual) < test_count:
                all_errors.append(
                    f"--strict: seed {seed}: only {len(actual)} prediction "
                    f"files; protocol requires at least the {test_count}-persona "
                    f"test split (see configs/splits.yaml)."
                )

    if all_errors:
        for line in all_errors[:50]:
            print(line, file=sys.stderr)
        if len(all_errors) > 50:
            print(
                f"... and {len(all_errors) - 50} more error(s)",
                file=sys.stderr,
            )
        return 1

    n_files = len(files)
    n_personas = sum(len(v) for v in personas_per_seed.values())
    print(
        f"OK: {n_files} files validated across {len(seeds_seen)} seed(s) "
        f"({n_personas} (seed, persona) tuples)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
