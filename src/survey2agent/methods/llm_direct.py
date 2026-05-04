"""LLM-Direct method (T3): forward an unmodified LLM answer.

`LLMDirect` is the forced-mode variant: `would_skip` from the LLM is
discarded and the model's raw label is returned as the final answer.
`LLMDirectSelective` honors the LLM's `would_skip` flag.

Both variants delegate the actual lookup to an `LLMSource`, so they work
identically against frozen JSON or future live dispatch.
"""

from __future__ import annotations

from survey2agent.extraction.atoms import ExtractedAtom

from .base import Method, Prediction
from .llm_base import LLMSource, normalize_to_prediction


class LLMDirect(Method):
    """Forced-mode: always return the LLM's raw label."""

    requires_fit = False
    requires_calibration = False

    def __init__(self, *, source: LLMSource, model_display_name: str) -> None:
        self.source = source
        self.model_display_name = model_display_name
        self.name = f"{model_display_name} Direct"

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        raw = self.source.get(atom.persona, qid)
        return Prediction(answer=raw.answer, would_skip=False, raw_answer=raw.answer)


class LLMDirectSelective(LLMDirect):
    """Selective: honor the LLM's `would_skip` flag."""

    def __init__(self, *, source: LLMSource, model_display_name: str) -> None:
        super().__init__(source=source, model_display_name=model_display_name)
        self.name = f"{model_display_name} Direct (Selective)"

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        raw = self.source.get(atom.persona, qid)
        return normalize_to_prediction(raw, qid)
