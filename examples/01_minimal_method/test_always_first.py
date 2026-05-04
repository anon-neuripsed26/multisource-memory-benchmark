"""Conformance test for the `AlwaysFirst` example method.

This is the minimum-viable test every new method should ship with:
instantiate the method, run it across all 18 questions on a fixture
atom, and check every return is a valid `Prediction`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS
from survey2agent.methods.base import Prediction

# The example directory begins with a digit, so it is not importable as a
# regular package. Load `always_first.py` by file path instead.
_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "always_first", _HERE / "always_first.py"
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules["always_first"] = _module
_spec.loader.exec_module(_module)
AlwaysFirst = _module.AlwaysFirst


@pytest.fixture
def empty_atom() -> ExtractedAtom:
    """A minimal valid atom: every (qid, source) slot is None."""
    return ExtractedAtom(
        persona="demo",
        extraction={qid: {} for qid in QUESTIONS},
    )


def test_always_first_returns_first_label_for_every_question(empty_atom: ExtractedAtom) -> None:
    method = AlwaysFirst()
    for qid, spec in QUESTIONS.items():
        pred = method.predict_one(empty_atom, qid)
        assert isinstance(pred, Prediction)
        assert pred.answer == spec["answer_space"][0]
        assert pred.would_skip is False
