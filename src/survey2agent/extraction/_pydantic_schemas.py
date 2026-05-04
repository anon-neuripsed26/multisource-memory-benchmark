"""Pydantic response schemas for structured LLM output.

Two families of schemas are exposed:

* :func:`build_extraction_response_model` — legacy per-source extraction. The
  LLM receives a source's NL text plus ``qids`` and must return
  ``{qid: Literal[allowed_labels] | None}``.

* :func:`build_persona_extraction_response_model` — released persona-level
  extraction. The LLM receives one persona's five rendered memory streams and
  returns ``{qid: {source: Literal[allowed_labels] | None}}`` for the full
  18-question × 5-source atom grid.

* :func:`build_answers_response_model` — whole-persona answer bundles for
  the LLM-Direct / Schema-Aware / Struct-LLM producers. The returned shape
  is ``{answers: {qid: {answer: str, would_skip: bool}}}``; the answer is
  a free-form string because the model is expected to predict the
  canonical label by name.

Both builders return a cacheable ``BaseModel`` subclass per unique qid list.
A small module-level cache keeps the generated class identity stable so
cache keys computed from ``response_schema`` remain deterministic across
calls (see ``CompletionRequest.to_cache_payload``).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

from .question_spec import QUESTIONS


def _literal_or_str(labels: tuple[str, ...]) -> Any:
    """Build a ``Literal[...]`` type from a label tuple, or ``str`` if empty."""
    if not labels:
        return str
    # `Literal` does not accept unpacking of dynamic tuples via Python < 3.11
    # in all cases; constructing via indexing is the standard workaround.
    return Literal[labels]  # type: ignore[valid-type]


@lru_cache(maxsize=256)
def build_extraction_response_model(
    qids: tuple[str, ...],
    source: str,
) -> Type[BaseModel]:
    """Return a dynamic Pydantic model for one source's extraction response.

    The model's fields are exactly ``qids``, each typed
    ``Optional[Literal[allowed_labels_for_qid]]``. Extra fields are rejected.

    Args:
        qids: Ordered tuple of question ids this source must answer.
        source: Source name (used only to disambiguate cached model classes;
            label whitelists are the same across sources).

    Raises:
        KeyError: if a qid is not present in ``QUESTIONS``.
    """
    fields: dict[str, tuple[Any, Any]] = {}
    for qid in qids:
        labels = tuple(QUESTIONS[qid]["answer_space"])
        tp = Optional[_literal_or_str(labels)]
        fields[qid] = (tp, Field(default=None))
    model = create_model(  # type: ignore[call-overload]
        f"ExtractionResponse_{source}_{len(qids)}",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )
    return model


@lru_cache(maxsize=32)
def build_persona_extraction_response_model(
    qids: tuple[str, ...],
    sources: tuple[str, ...],
) -> Type[BaseModel]:
    """Return a dynamic Pydantic model for one persona's full atom grid.

    Shape: ``{qid: {source: Optional[Literal[allowed_labels_for_qid]]}}``.
    Extra qids or source keys are rejected. The label whitelist is tied to the
    question id, so each qid gets its own nested source model.
    """
    outer_fields: dict[str, tuple[Any, Any]] = {}
    for qid in qids:
        labels = tuple(QUESTIONS[qid]["answer_space"])
        tp = Optional[_literal_or_str(labels)]
        source_fields: dict[str, tuple[Any, Any]] = {
            source: (tp, Field(default=None)) for source in sources
        }
        source_model = create_model(  # type: ignore[call-overload]
            f"ExtractionSources_{qid}_{len(sources)}",
            __config__=ConfigDict(extra="forbid"),
            **source_fields,
        )
        outer_fields[qid] = (source_model, Field(...))

    model = create_model(  # type: ignore[call-overload]
        f"PersonaExtractionResponse_{len(qids)}_{len(sources)}",
        __config__=ConfigDict(extra="forbid"),
        **outer_fields,
    )
    return model


class _AnswerEntry(BaseModel):
    """One persona-question entry in an answers bundle."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    would_skip: bool = False


@lru_cache(maxsize=32)
def build_answers_response_model(qids: tuple[str, ...]) -> Type[BaseModel]:
    """Return a dynamic Pydantic model for a whole-persona answers response.

    Shape: ``{answers: {qid: {answer: str, would_skip: bool}}}``.

    Each qid maps to a nested ``_AnswerEntry``; the outer ``answers`` field
    is a ``dict[qid, _AnswerEntry]`` with extra qids rejected by a root
    validator (Pydantic v2 ``model_validator``).
    """
    # Build a dedicated inner model whose keys are exactly the qid set.
    # Using a nested dynamic model keeps JSON Schema emission clean for
    # providers that require a fully closed schema (OpenAI json_schema
    # strict=True).
    inner_fields: dict[str, tuple[Any, Any]] = {
        qid: (_AnswerEntry, Field(...)) for qid in qids
    }
    inner_model = create_model(  # type: ignore[call-overload]
        f"AnswersInner_{len(qids)}",
        __config__=ConfigDict(extra="forbid"),
        **inner_fields,
    )

    outer_fields: dict[str, tuple[Any, Any]] = {
        "answers": (inner_model, Field(...)),
    }
    outer_model = create_model(  # type: ignore[call-overload]
        f"AnswersResponse_{len(qids)}",
        __config__=ConfigDict(extra="forbid"),
        **outer_fields,
    )
    return outer_model


__all__ = [
    "build_extraction_response_model",
    "build_persona_extraction_response_model",
    "build_answers_response_model",
]
