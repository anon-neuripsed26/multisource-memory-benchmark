"""Tests for the Hugging Face bundle fetch helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.fetch_benchmark import REQUIRED_PATHS, validate_bundle_layout


def _materialize_required_layout(root: Path) -> None:
    for rel in REQUIRED_PATHS:
        path = root / rel
        if "." in Path(rel).name:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder\n", encoding="utf-8")
        else:
            path.mkdir(parents=True, exist_ok=True)


def test_validate_bundle_layout_fails_on_missing_paths(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        validate_bundle_layout(tmp_path)
    assert "Download incomplete" in str(exc.value)
    assert "benchmark/seeds" in str(exc.value)


def test_validate_bundle_layout_accepts_expected_hf_tree(tmp_path: Path) -> None:
    _materialize_required_layout(tmp_path)
    validate_bundle_layout(tmp_path)
