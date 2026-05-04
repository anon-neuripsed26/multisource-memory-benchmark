"""LLM-Few-Shot method (T3): LLM prompted with cached training-set exemplars.

Few-shot is `requires_fit=True` because the in-context exemplars are an
artifact derived from the train split. In this frozen-validation phase
the exemplars and the resulting per-(persona, qid) answers are already
materialized on disk; `fit` therefore only validates that the manifest
exists and records the train size for `state_dict` parity. Live shot
selection is deferred to a future `scripts/regen_llm_frozen.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from survey2agent.extraction.atoms import ExtractedAtom

from .base import Method, Prediction, TrainingRecord
from .llm_base import LLMSource, normalize_to_prediction


class LLMFewShot(Method):
    """Forced-mode few-shot variant."""

    requires_fit = True
    requires_calibration = False

    def __init__(
        self,
        *,
        source: LLMSource,
        model_display_name: str,
        shot_manifest_path: Path | None = None,
    ) -> None:
        self.source = source
        self.model_display_name = model_display_name
        self.shot_manifest_path = (
            Path(shot_manifest_path) if shot_manifest_path is not None else None
        )
        self.name = f"{model_display_name} Few-Shot"
        self._train_size: int = 0
        self._fitted: bool = False

    def fit(self, records: Sequence[TrainingRecord]) -> None:
        """Validate the shot manifest (if provided) and record train size.

        No exemplar regeneration is performed — frozen mode assumes the
        underlying source already encodes the cached few-shot answers.
        """
        if self.shot_manifest_path is not None and not self.shot_manifest_path.exists():
            raise FileNotFoundError(
                f"Few-shot manifest not found: {self.shot_manifest_path}"
            )
        self._train_size = len(records)
        self._fitted = True

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        raw = self.source.get(atom.persona, qid)
        return Prediction(answer=raw.answer, would_skip=False, raw_answer=raw.answer)

    def state_dict(self) -> dict:
        return {
            "model_display_name": self.model_display_name,
            "shot_manifest_path": (
                str(self.shot_manifest_path) if self.shot_manifest_path else None
            ),
            "train_size": self._train_size,
            "fitted": self._fitted,
        }

    def load_state_dict(self, state: dict) -> None:
        self._train_size = int(state.get("train_size", 0))
        self._fitted = bool(state.get("fitted", False))


class LLMFewShotSelective(LLMFewShot):
    """Selective few-shot variant."""

    def __init__(
        self,
        *,
        source: LLMSource,
        model_display_name: str,
        shot_manifest_path: Path | None = None,
    ) -> None:
        super().__init__(
            source=source,
            model_display_name=model_display_name,
            shot_manifest_path=shot_manifest_path,
        )
        self.name = f"{model_display_name} Few-Shot (Selective)"

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        raw = self.source.get(atom.persona, qid)
        return normalize_to_prediction(raw, qid)
