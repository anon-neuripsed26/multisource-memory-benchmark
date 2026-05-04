"""Shared abstractions for LLM-based methods (Direct, Schema-Aware, Few-Shot).

An `LLMSource` is anything that can return a `RawLLMOutput` for a
`(persona_id, qid)` query. Concrete sources live in `llm_sources.py`:
the frozen-JSON loaders read pre-computed model outputs from disk so the
test runs are byte-deterministic and require no network calls.

`normalize_to_prediction` converts a raw LLM output into the canonical
`Prediction` form used by the runner, preserving the model's underlying
label in `Prediction.raw_answer` even when the selective layer turns the
prediction into a SKIP (so the forced-mode evaluator can still score it).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .base import SKIP_SENTINEL, Prediction


@dataclass(frozen=True)
class RawLLMOutput:
    """A single (persona, qid) output as the LLM emitted it."""

    answer: str
    would_skip: bool


class LLMSource(ABC):
    """Resolver from `(persona_id, qid)` to the LLM's raw answer."""

    @abstractmethod
    def get(self, persona_id: str, qid: str) -> RawLLMOutput:
        """Return the LLM's raw output for one persona/question pair."""


def normalize_to_prediction(raw: RawLLMOutput, qid: str) -> Prediction:
    """Convert `RawLLMOutput` to `Prediction`, preserving the raw label.

    `qid` is currently unused but accepted for symmetry with `predict_one`
    and to allow future qid-conditional normalization without an API change.
    """
    del qid  # reserved for future use
    if raw.would_skip:
        return Prediction(answer=SKIP_SENTINEL, would_skip=True, raw_answer=raw.answer)
    return Prediction(answer=raw.answer, would_skip=False, raw_answer=raw.answer)
