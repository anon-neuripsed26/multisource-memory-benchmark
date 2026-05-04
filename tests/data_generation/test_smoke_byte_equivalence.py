"""Byte-equivalent smoke test for the ``data_generation`` package.

Invariant under test
--------------------
The migrated package ``survey2agent.data_generation.*`` must produce
**byte-identical** outputs to the published reference dataset at
``$DATA_ROOT/benchmark/seeds/s20260321/`` for the same seed,
across the full pipeline:
``L1 personas -> L2 events -> L3 sources -> L4 ground truth``.

This is the correct invariant for the *paper* artifact: anyone who runs
``survey2agent.data_generation`` with seed 20260321 must reproduce the
exact dataset used in the experiments.

Note on schema lock
-------------------
This invariant requires that ``SparsityInjection`` includes the legacy
``missing_field_prob`` field. That field is sampled but never consumed
(see ``SparsityInjection`` docstring for the rationale). Removing it shifts the numpy RNG sequence
for the ``stated_vs_revealed`` track, breaking byte-level reproducibility
of all 160 stated personas.

Subset strategy
---------------
* L1 (personas.json) is generated for the full 480 personas.
* L2 / L3 / L4 are restricted to the 10 alphabetically-first persona IDs
  via each script's ``--persona`` flag; each persona uses a deterministic
  per-persona seed and is independent of the others.
"""

from __future__ import annotations

import filecmp
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Paths -----------------------------------------------------------------------
# Test file path: tests/data_generation/test_*.py
# parents[0]=data_generation, [1]=tests, [2]=repo root
from survey2agent._paths import seed_dir as _seed_dir
ON_DISK_REFERENCE = _seed_dir("s20260321")
REPO_ROOT = Path(__file__).resolve().parents[2]

PYTHON = sys.executable

pytestmark = pytest.mark.needs_data
SEED = 20260321
N_SMOKE = 10
# The on-disk dataset is laid out at ``v1.0 reference``,
# but the ``benchmark_metadata.version`` field embedded inside ``personas.json`` is
# ``test_run_v2_p480_s20260321`` (the directory was renamed after generation, dropping
# the ``v2_`` infix). Because the version field is set from the output-dir basename,
# we must generate into a dir named with the ``v2_`` form to reproduce the embedded
# version, then compare against the (renamed) on-disk path.
GENERATION_DIR_NAME = "test_run_v2_p480_s20260321"  # drives benchmark_metadata.version

SOURCE_FILES = [
    "objective_log.json",
    "device_log.json",
    "planner.json",
    "daily_self_report.json",
    "profile_ltm.json",
]


def _run_module(module: str, args: list[str], cwd: Path, extra_pythonpath: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(extra_pythonpath) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [PYTHON, "-m", module, *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"`python -m {module}` failed (rc={result.returncode}):\n"
            f"  args: {args}\n  stdout:\n{result.stdout}\n  stderr:\n{result.stderr}"
        )


def _generate_full_pipeline(out_root: Path) -> list[str]:
    """Run L1 (full 480) then L2/L3/L4 for the first N_SMOKE persona IDs."""
    cwd = REPO_ROOT
    pythonpath = cwd / "src"

    _run_module(
        "survey2agent.data_generation.generate_personas",
        ["--seed", str(SEED), "--output-dir", str(out_root)],
        cwd=cwd,
        extra_pythonpath=pythonpath,
    )

    personas_path = out_root / "config" / "personas.json"
    personas_data = json.loads(personas_path.read_text(encoding="utf-8"))
    smoke_ids = sorted(p["id"] for p in personas_data["personas"])[:N_SMOKE]

    for pid in smoke_ids:
        for sub in ("generate_events", "generate_sources", "generate_ground_truth"):
            _run_module(
                f"survey2agent.data_generation.{sub}",
                ["--dataset-dir", str(out_root), "--persona", pid],
                cwd=cwd,
                extra_pythonpath=pythonpath,
            )
    return smoke_ids


@pytest.fixture(scope="module")
def fresh_dataset(tmp_path_factory) -> tuple[Path, list[str]]:
    """Generate the published-dataset layout from the migrated package.

    The output dir basename is locked to ``test_run_p480_s20260321`` so
    that ``benchmark_metadata.version`` (set to the basename) matches the
    on-disk reference.
    """
    base = tmp_path_factory.mktemp("data_gen_smoke")
    out_root = base / GENERATION_DIR_NAME
    smoke_ids = _generate_full_pipeline(out_root)
    return out_root, smoke_ids


def _diff(fresh: Path, reference: Path, label: str) -> str | None:
    if not fresh.exists():
        return f"{label}: fresh missing ({fresh})"
    if not reference.exists():
        return f"{label}: reference missing ({reference})"
    if filecmp.cmp(str(fresh), str(reference), shallow=False):
        return None
    return (
        f"{label}: bytes differ "
        f"(size_fresh={fresh.stat().st_size} size_reference={reference.stat().st_size})"
    )


@pytest.mark.skipif(
    not ON_DISK_REFERENCE.exists(),
    reason="on-disk reference dataset not present",
)
def test_personas_json_byte_equivalent_to_reference(fresh_dataset) -> None:
    """L1: personas.json must byte-match the published dataset."""
    fresh_root, _ = fresh_dataset
    diff = _diff(
        fresh_root / "config" / "personas.json",
        ON_DISK_REFERENCE / "config" / "personas.json",
        "config/personas.json",
    )
    assert diff is None, diff


@pytest.mark.skipif(
    not ON_DISK_REFERENCE.exists(),
    reason="on-disk reference dataset not present",
)
def test_event_tables_byte_equivalent_to_reference(fresh_dataset) -> None:
    """L2: event_table.json for the first 10 personas must byte-match."""
    fresh_root, smoke_ids = fresh_dataset
    failures = []
    for pid in smoke_ids:
        diff = _diff(
            fresh_root / pid / "event_table.json",
            ON_DISK_REFERENCE / pid / "event_table.json",
            f"{pid}/event_table.json",
        )
        if diff:
            failures.append(diff)
    assert not failures, "\n".join(failures)


@pytest.mark.skipif(
    not ON_DISK_REFERENCE.exists(),
    reason="on-disk reference dataset not present",
)
def test_sources_byte_equivalent_to_reference(fresh_dataset) -> None:
    """L3: all 5 structural sources for the first 10 personas must byte-match."""
    fresh_root, smoke_ids = fresh_dataset
    failures = []
    for pid in smoke_ids:
        for fname in SOURCE_FILES:
            diff = _diff(
                fresh_root / pid / "structural_sources" / fname,
                ON_DISK_REFERENCE / pid / "structural_sources" / fname,
                f"{pid}/structural_sources/{fname}",
            )
            if diff:
                failures.append(diff)
    assert not failures, "\n".join(failures)


@pytest.mark.skipif(
    not ON_DISK_REFERENCE.exists(),
    reason="on-disk reference dataset not present",
)
def test_ground_truth_byte_equivalent_to_reference(fresh_dataset) -> None:
    """L4: ground_truth.json for the first 10 personas must byte-match."""
    fresh_root, smoke_ids = fresh_dataset
    failures = []
    for pid in smoke_ids:
        diff = _diff(
            fresh_root / pid / "ground_truth.json",
            ON_DISK_REFERENCE / pid / "ground_truth.json",
            f"{pid}/ground_truth.json",
        )
        if diff:
            failures.append(diff)
    assert not failures, "\n".join(failures)


@pytest.mark.skipif(
    not ON_DISK_REFERENCE.exists(),
    reason="on-disk reference dataset not present",
)
def test_stated_track_byte_equivalent_to_reference(fresh_dataset) -> None:
    """Targeted check: at least one ``bench_stated_*`` persona's full file
    set is byte-equivalent. The stated track is the one most sensitive to
    the ``SparsityInjection.missing_field_prob`` schema lock; this test
    guards against any future schema regression that would break paper
    reproducibility for the 160-persona stated-vs-revealed track.
    """
    fresh_root, _ = fresh_dataset
    personas_data = json.loads(
        (fresh_root / "config" / "personas.json").read_text(encoding="utf-8")
    )
    stated_ids = sorted(
        p["id"] for p in personas_data["personas"] if p["id"].startswith("bench_stated_")
    )
    if not stated_ids:
        pytest.skip("no bench_stated_* personas in the L1 output")

    target = stated_ids[0]
    # Generate L2/L3/L4 for this stated persona.
    cwd = REPO_ROOT
    pythonpath = cwd / "src"
    for sub in ("generate_events", "generate_sources", "generate_ground_truth"):
        _run_module(
            f"survey2agent.data_generation.{sub}",
            ["--dataset-dir", str(fresh_root), "--persona", target],
            cwd=cwd,
            extra_pythonpath=pythonpath,
        )

    failures = []
    files_to_check = [
        f"{target}/event_table.json",
        f"{target}/ground_truth.json",
        *(f"{target}/structural_sources/{f}" for f in SOURCE_FILES),
    ]
    for rel in files_to_check:
        diff = _diff(fresh_root / rel, ON_DISK_REFERENCE / rel, rel)
        if diff:
            failures.append(diff)
    assert not failures, "\n".join(failures)
