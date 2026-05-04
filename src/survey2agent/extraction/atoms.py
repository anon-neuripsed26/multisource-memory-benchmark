"""Frozen dataclasses + readers for extracted atoms and method predictions.

Two on-disk JSON schemas are supported:

1. Atom schema (`ExtractedAtom`):
       {"persona": "...", "extraction": {<qid>: {<source>: <enum_str|None>}}}

2. Method-prediction schema (`MethodPrediction`):
       {"persona": "...", "answers": {<qid>: {"answer": <str>, "would_skip": <bool>}}}

Both classes are immutable: the dataclass is `frozen`, and every nested mapping
is wrapped in `types.MappingProxyType` so deep mutation raises `TypeError`.
Construct instances via the `from_json` classmethods (which validate schema)
or via the directory loaders below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

EXPECTED_QUESTION_IDS: tuple[str, ...] = (
    "A1", "A2", "A3",
    "B2", "B3",
    "C2", "C3",
    "D1", "D2",
    "E1", "E2",
    "F1", "F2", "F3",
    "G1", "G2",
    "Ctrl1", "Ctrl2",
)

EXPECTED_SOURCES: tuple[str, ...] = (
    "profile_ltm",
    "planner",
    "daily_self_report",
    "objective_log",
    "device_log",
)


def _freeze_extraction(
    raw: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, str | None]]:
    """Wrap each per-qid source dict in MappingProxyType, then wrap the outer."""
    inner: dict[str, Mapping[str, str | None]] = {}
    for qid, source_map in raw.items():
        inner[qid] = MappingProxyType(dict(source_map))
    return MappingProxyType(inner)


def _freeze_answers(
    raw: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Mapping[str, str | bool]]:
    inner: dict[str, Mapping[str, str | bool]] = {}
    for qid, answer_map in raw.items():
        inner[qid] = MappingProxyType(dict(answer_map))
    return MappingProxyType(inner)


@dataclass(frozen=True)
class ExtractedAtom:
    """Per-persona structured extraction `μ̂` from NL memory.

    `extraction[qid][source]` holds an enum-string answer or `None` if the
    source did not provide an answer for that question.
    """

    persona: str
    extraction: Mapping[str, Mapping[str, str | None]]

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "ExtractedAtom":
        if "persona" not in raw or not isinstance(raw["persona"], str):
            raise ValueError("atom JSON missing string 'persona' field")
        persona = raw["persona"]
        if "extraction" not in raw or not isinstance(raw["extraction"], dict):
            raise ValueError(f"atom JSON missing dict 'extraction' for persona {persona}")
        extraction = raw["extraction"]
        for qid in EXPECTED_QUESTION_IDS:
            if qid not in extraction:
                raise ValueError(f"missing question {qid} in persona {persona}")
            source_map = extraction[qid]
            if not isinstance(source_map, dict):
                raise ValueError(
                    f"extraction[{qid}] must be a dict in persona {persona}, got {type(source_map).__name__}"
                )
            for src in EXPECTED_SOURCES:
                if src not in source_map:
                    raise ValueError(
                        f"missing source {src!r} for question {qid} in persona {persona}"
                    )
                val = source_map[src]
                if val == "null":
                    source_map[src] = None
                    val = None
                if val is not None and not isinstance(val, str):
                    raise ValueError(
                        f"extraction[{qid}][{src}] must be str or None in persona {persona}, "
                        f"got {type(val).__name__}"
                    )
        return cls(persona=persona, extraction=_freeze_extraction(extraction))


@dataclass(frozen=True)
class MethodPrediction:
    """Per-persona method output: an answer (+ skip flag) per question."""

    persona: str
    answers: Mapping[str, Mapping[str, str | bool]]

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> "MethodPrediction":
        if "persona" not in raw or not isinstance(raw["persona"], str):
            raise ValueError("method-prediction JSON missing string 'persona' field")
        persona = raw["persona"]
        if "answers" not in raw or not isinstance(raw["answers"], dict):
            raise ValueError(
                f"method-prediction JSON missing dict 'answers' for persona {persona}"
            )
        answers = raw["answers"]
        for qid in EXPECTED_QUESTION_IDS:
            if qid not in answers:
                raise ValueError(f"missing question {qid} in persona {persona}")
            entry = answers[qid]
            if not isinstance(entry, dict):
                raise ValueError(
                    f"answers[{qid}] must be a dict in persona {persona}, got {type(entry).__name__}"
                )
            if "answer" not in entry:
                raise ValueError(f"answers[{qid}] missing 'answer' field in persona {persona}")
            if "would_skip" not in entry:
                raise ValueError(f"answers[{qid}] missing 'would_skip' field in persona {persona}")
            if not isinstance(entry["answer"], str):
                raise ValueError(
                    f"answers[{qid}]['answer'] must be str in persona {persona}, "
                    f"got {type(entry['answer']).__name__}"
                )
            if not isinstance(entry["would_skip"], bool):
                raise ValueError(
                    f"answers[{qid}]['would_skip'] must be bool in persona {persona}, "
                    f"got {type(entry['would_skip']).__name__}"
                )
        return cls(persona=persona, answers=_freeze_answers(answers))


def load_atom(path: Path) -> ExtractedAtom:
    """Load and validate a single extracted-atom JSON file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return ExtractedAtom.from_json(raw)


def load_atoms_from_dir(directory: Path) -> dict[str, ExtractedAtom]:
    """Load all `*.json` atom files from `directory`. Returns {persona_id: atom}."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"not a directory: {directory}")
    out: dict[str, ExtractedAtom] = {}
    for path in sorted(directory.glob("*.json")):
        atom = load_atom(path)
        if atom.persona in out:
            raise ValueError(f"duplicate persona {atom.persona} in {directory}")
        out[atom.persona] = atom
    return out


def load_method_prediction(path: Path) -> MethodPrediction:
    """Load and validate a single method-prediction JSON file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return MethodPrediction.from_json(raw)


def load_method_predictions_from_dir(directory: Path) -> dict[str, MethodPrediction]:
    """Load all `*.json` method-prediction files. Returns {persona_id: prediction}."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"not a directory: {directory}")
    out: dict[str, MethodPrediction] = {}
    for path in sorted(directory.glob("*.json")):
        pred = load_method_prediction(path)
        if pred.persona in out:
            raise ValueError(f"duplicate persona {pred.persona} in {directory}")
        out[pred.persona] = pred
    return out
