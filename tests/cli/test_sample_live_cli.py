"""Offline checks for the documented sample live-call command.

These tests intentionally do not hit any provider API. They verify that the
sample persona/question selectors, prompt bundle, and CLI path resolve cleanly
up to the expected cache miss or missing-key boundary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_ROOT = REPO_ROOT / "data" / "sample"


def _run_sample_few_shot(tmp_path: Path, *, allow_api_call: bool) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "survey2agent",
        "run-few-shot",
        "--provider",
        "openrouter",
        "--model",
        "deepseek-v3.2",
        "--seed",
        str(SAMPLE_ROOT / "benchmark" / "seeds" / "s20260321"),
        "--personas",
        str(SAMPLE_ROOT / "personas_one.txt"),
        "--questions",
        str(SAMPLE_ROOT / "questions_one.txt"),
        "--configs-root",
        str(REPO_ROOT / "configs" / "few_shot"),
        "--output-dir",
        str(tmp_path / "out"),
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    if allow_api_call:
        cmd.append("--allow-api-call")

    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_sample_selectors_exist_and_point_to_fixture() -> None:
    assert (SAMPLE_ROOT / "personas_one.txt").read_text(encoding="utf-8").strip() == (
        "bench_shift_121_avery_ellis"
    )
    assert (SAMPLE_ROOT / "questions_one.txt").read_text(encoding="utf-8").strip() == "A1"
    assert (
        SAMPLE_ROOT
        / "benchmark"
        / "seeds"
        / "s20260321"
        / "bench_shift_121_avery_ellis"
        / "structural_sources"
    ).is_dir()


def test_sample_few_shot_cache_only_reaches_expected_cache_miss(tmp_path: Path) -> None:
    result = _run_sample_few_shot(tmp_path, allow_api_call=False)
    assert result.returncode == 1
    assert "Running in cache-only mode" in result.stderr

    failures = json.loads((tmp_path / "out" / "_failures.json").read_text(encoding="utf-8"))
    assert list(failures) == ["bench_shift_121_avery_ellis__A1"]
    assert "cache miss" in failures["bench_shift_121_avery_ellis__A1"]


def test_sample_few_shot_live_mode_stops_at_missing_key_without_network(tmp_path: Path) -> None:
    result = _run_sample_few_shot(tmp_path, allow_api_call=True)
    assert result.returncode == 1

    failures = json.loads((tmp_path / "out" / "_failures.json").read_text(encoding="utf-8"))
    assert list(failures) == ["bench_shift_121_avery_ellis__A1"]
    assert "OPENROUTER_API_KEY environment variable is not set" in failures[
        "bench_shift_121_avery_ellis__A1"
    ]
