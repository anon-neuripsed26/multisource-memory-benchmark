"""Tests for the frozen artifact quirks patch table.

Validates that:
  1. ``frozen_artifact_quirks.json`` is well-formed and has the expected schema.
  2. The known Qwen3 Ctrl2 singleton is in the patch table.
  3. ``FrozenBulkJSONSource`` applies the patch when loading the affected
     (model, seed, variant, persona, qid) tuple.
  4. Other (qid) entries in the same persona file are untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from survey2agent._paths import METHOD_OUTPUTS_ROOT
from survey2agent.methods import FrozenBulkJSONSource

_BULK_ROOT = METHOD_OUTPUTS_ROOT
_QUIRKS_PATH = _BULK_ROOT / "frozen_artifact_quirks.json"

_QUIRK_KEY = {
    "model": "qwen3-235b-a22b-2507",
    "seed": "s20260321",
    "variant": "direct",
    "persona": "bench_stable_151_alex_carter",
    "qid": "Ctrl2",
}

pytestmark = pytest.mark.needs_data


def test_frozen_artifact_quirks_json_valid() -> None:
    assert _QUIRKS_PATH.exists(), f"missing {_QUIRKS_PATH}"
    data = json.loads(_QUIRKS_PATH.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1"
    assert "patches" in data
    assert isinstance(data["patches"], list)
    for patch in data["patches"]:
        for key in ("model", "seed", "variant", "persona", "qid", "patch", "reason"):
            assert key in patch, f"patch missing key {key!r}: {patch}"
        assert "answer" in patch["patch"]
        assert "would_skip" in patch["patch"]


def test_frozen_artifact_quirks_qwen3_singleton_present() -> None:
    data = json.loads(_QUIRKS_PATH.read_text(encoding="utf-8"))
    keys = {
        (p["model"], p["seed"], p["variant"], p["persona"], p["qid"])
        for p in data["patches"]
    }
    assert (
        _QUIRK_KEY["model"],
        _QUIRK_KEY["seed"],
        _QUIRK_KEY["variant"],
        _QUIRK_KEY["persona"],
        _QUIRK_KEY["qid"],
    ) in keys


@pytest.mark.needs_data
def test_frozen_source_applies_quirk() -> None:
    """The patched record must NOT come back as ``answer="SKIP"``.

    Original record on disk: ``{"answer": "SKIP", "would_skip": false}``
    (violates ``Prediction`` invariant). Quirk replaces ``answer`` with the
    sentinel ``"__QUIRK_SKIP_STRING__"`` while leaving ``would_skip=False``,
    mirroring legacy ``_compute_ds_qw_skip.py`` semantics (forced wrong
    answer rather than skip).
    """
    src = FrozenBulkJSONSource(
        _QUIRK_KEY["model"], _QUIRK_KEY["seed"], _QUIRK_KEY["variant"]
    )
    out = src.get(_QUIRK_KEY["persona"], _QUIRK_KEY["qid"])
    assert out.answer == "__QUIRK_SKIP_STRING__"
    assert out.would_skip is False


@pytest.mark.needs_data
def test_frozen_source_unaffected_records_untouched() -> None:
    """A non-patched qid in the same persona file is returned verbatim."""
    src = FrozenBulkJSONSource(
        _QUIRK_KEY["model"], _QUIRK_KEY["seed"], _QUIRK_KEY["variant"]
    )
    out = src.get(_QUIRK_KEY["persona"], "A1")
    raw = json.loads(
        (
            _BULK_ROOT
            / _QUIRK_KEY["model"]
            / _QUIRK_KEY["seed"]
            / _QUIRK_KEY["variant"]
            / f"{_QUIRK_KEY['persona']}.json"
        ).read_text(encoding="utf-8")
    )
    expected = raw["answers"]["A1"]
    assert out.answer == expected["answer"]
    assert out.would_skip == expected["would_skip"]


def test_frozen_source_no_quirk_when_root_lacks_file(tmp_path: Path) -> None:
    """Source built against a tmp root with no quirks file applies no patches."""
    variant_dir = tmp_path / "fake-model" / "s_test" / "direct"
    variant_dir.mkdir(parents=True)
    payload = {
        "persona": "p1",
        "answers": {"A1": {"answer": "Z", "would_skip": False}},
    }
    (variant_dir / "p1.json").write_text(json.dumps(payload), encoding="utf-8")
    src = FrozenBulkJSONSource("fake-model", "s_test", "direct", root=tmp_path)
    out = src.get("p1", "A1")
    assert out.answer == "Z"
    assert out.would_skip is False
