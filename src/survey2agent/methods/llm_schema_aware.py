"""LLM-Schema-Aware method (T3): LLM prompted with the answer-space schema.

Identical control flow to `LLMDirect` but uses a different `LLMSource`
(typically `FrozenBulkJSONSource` pointing at the `schema-aware` variant).
The two are kept as separate classes so that paper-facing `name` strings
are unambiguous and so that `requires_fit` / prompt-hash bookkeeping can
diverge in future without an interface change.
"""

from __future__ import annotations

from survey2agent.extraction.atoms import ExtractedAtom

from .base import Method, Prediction
from .llm_base import LLMSource, normalize_to_prediction


class LLMSchemaAware(Method):
    """Forced-mode schema-aware variant."""

    requires_fit = False
    requires_calibration = False

    def __init__(self, *, source: LLMSource, model_display_name: str) -> None:
        self.source = source
        self.model_display_name = model_display_name
        self.name = f"{model_display_name} Schema"

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        raw = self.source.get(atom.persona, qid)
        return Prediction(answer=raw.answer, would_skip=False, raw_answer=raw.answer)


class LLMSchemaAwareSelective(LLMSchemaAware):
    """Selective schema-aware variant."""

    def __init__(self, *, source: LLMSource, model_display_name: str) -> None:
        super().__init__(source=source, model_display_name=model_display_name)
        self.name = f"{model_display_name} Schema (Selective)"

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        raw = self.source.get(atom.persona, qid)
        return normalize_to_prediction(raw, qid)
