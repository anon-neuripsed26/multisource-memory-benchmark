"""Adapter: ExtractedAtom → per-question mu_q dict (method input).

Method classes (T0/T1/T2 baselines, ABF, oracle) consume a per-question dict
``mu_q: dict[str, str | None]`` keyed by source name, where ``None`` means
the source produced no value for that question. The on-disk atom JSON is
already in that exact shape under ``extraction[qid]``; this adapter performs
a defensive copy (the ExtractedAtom wraps each per-qid dict in a read-only
``MappingProxyType``) so downstream code receives a plain mutable ``dict``.
"""

from __future__ import annotations

from typing import Iterator

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS, SOURCE_NAMES


def atom_to_mu_q(atom: ExtractedAtom, qid: str) -> dict[str, str | None]:
    """Return ``{source_name: extracted_value_or_None}`` for one question.

    Key order follows ``SOURCE_NAMES``. ``None`` values pass through
    unchanged (they are not coerced to the string ``"null"`` or dropped).

    Raises:
        KeyError: if ``qid`` is unknown or the atom is missing a source.
    """
    if qid not in QUESTIONS:
        raise KeyError(f"unknown question id: {qid!r}")
    source_map = atom.extraction[qid]
    return {source: source_map[source] for source in SOURCE_NAMES}


def iter_mu_q_per_question(
    atom: ExtractedAtom,
) -> Iterator[tuple[str, dict[str, str | None]]]:
    """Yield ``(qid, mu_q)`` for every question in the canonical order."""
    for qid in QUESTIONS:
        yield qid, atom_to_mu_q(atom, qid)
