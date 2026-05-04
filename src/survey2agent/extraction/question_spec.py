"""Question schema for the released benchmark (18 questions × 5 sources).

The authoritative machine-readable spec lives at
`configs/questions.yaml`. This module is a thin loader that
parses the YAML once at import time and exposes the same public API the
rest of the codebase already consumes:

    QUESTIONS       : dict[qid, dict[...]]   - per-question schema
    QUESTION_TEXT   : dict[qid, str]         - NL prompt strings
    SOURCE_NAMES    : tuple[str, ...]        - canonical source order
    BIAS_DEFAULTS   : dict[source, int|dict] - QBD-2 bias matrix (raw)
    BIAS_OVERRIDES  : dict[qid, dict[source, int]] - per-question overrides
    get_bias(qid)   : dict[source, int]      - per-question bias resolved
                                               against QUESTIONS[qid]["topic"]
                                               and BIAS_OVERRIDES[qid]

The narrative spec lives at
`v1.0 reference`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Resolve YAML path:
#   src/survey2agent/extraction/question_spec.py
#   parents[0] = extraction
#   parents[1] = survey2agent
#   parents[2] = src
#   parents[3] = release repo root
_YAML_PATH: Path = (
    Path(__file__).resolve().parents[3] / "configs" / "questions.yaml"
)


def _load_spec() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, str],
    tuple[str, ...],
    dict[str, Any],
    dict[str, dict[str, int]],
]:
    with _YAML_PATH.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    source_names: tuple[str, ...] = tuple(raw["source_names"])
    bias_defaults: dict[str, Any] = dict(raw.get("bias_defaults", {}))
    bias_overrides_raw: dict[str, Any] = dict(raw.get("bias_overrides", {}) or {})

    questions: dict[str, dict[str, Any]] = {}
    question_text: dict[str, str] = {}

    for qid, entry in raw["questions"].items():
        question_text[qid] = entry["question_text"]

        ordered_labels = entry.get("ordered_labels")
        if ordered_labels is not None:
            ordinal_encoding: dict[str, int] | None = {
                label: i + 1 for i, label in enumerate(ordered_labels)
            }
        else:
            ordinal_encoding = None

        questions[qid] = {
            "type": entry["type"],
            "topic": entry["topic"],
            "answer_space": list(entry["answer_space"]),
            "answer_space_type": entry["answer_space_type"],
            "ordinal_encoding": ordinal_encoding,
            "edge_options": list(entry["edge_options"]),
            "time_window": entry["time_window"],
        }

    # Validate bias_overrides shape against the loaded question + source set.
    bias_overrides: dict[str, dict[str, int]] = {}
    for qid, per_q in bias_overrides_raw.items():
        if qid not in questions:
            raise KeyError(
                f"bias_overrides references unknown question id: {qid!r}"
            )
        if not isinstance(per_q, dict):
            raise TypeError(
                f"bias_overrides[{qid!r}] must be a mapping of "
                f"source -> int, got {type(per_q).__name__}"
            )
        coerced: dict[str, int] = {}
        for source, value in per_q.items():
            if source not in source_names:
                raise KeyError(
                    f"bias_overrides[{qid!r}] references unknown source: "
                    f"{source!r} (must be one of {source_names})"
                )
            coerced[source] = int(value)
        bias_overrides[qid] = coerced

    return (
        questions,
        question_text,
        source_names,
        bias_defaults,
        bias_overrides,
    )


(
    QUESTIONS,
    QUESTION_TEXT,
    SOURCE_NAMES,
    BIAS_DEFAULTS,
    BIAS_OVERRIDES,
) = _load_spec()


def get_bias(qid: str) -> dict[str, int]:
    """Return per-source bias `b_{s, topic(qid)}` as a `{source: int}` dict.

    Resolution order (per source):

        1. ``BIAS_OVERRIDES[qid][source]`` if present (per-question override).
        2. ``BIAS_DEFAULTS[source]``: if the entry is a topic-keyed dict, use
           ``BIAS_DEFAULTS[source][topic]`` where ``topic = QUESTIONS[qid]
           ["topic"]`` (per QBD-2). If the entry is a scalar, use it directly.
        3. ``0`` if the source is missing from ``BIAS_DEFAULTS``.

    Per-question overrides may be partial: any source not listed in
    ``BIAS_OVERRIDES[qid]`` falls back to the topic-level default.

    Raises:
        KeyError: if `qid` is not a known question id, or if a source has a
            topic-dependent default but the question's topic is missing from
            the mapping.
    """
    if qid not in QUESTIONS:
        raise KeyError(f"unknown question id: {qid!r}")
    topic = QUESTIONS[qid]["topic"]
    overrides = BIAS_OVERRIDES.get(qid, {})

    out: dict[str, int] = {}
    for source in SOURCE_NAMES:
        if source in overrides:
            out[source] = overrides[source]
            continue
        spec = BIAS_DEFAULTS.get(source, 0)
        if isinstance(spec, dict):
            if topic not in spec:
                raise KeyError(
                    f"bias_defaults[{source!r}] missing topic {topic!r} "
                    f"required by question {qid!r}"
                )
            out[source] = int(spec[topic])
        else:
            out[source] = int(spec)
    return out
