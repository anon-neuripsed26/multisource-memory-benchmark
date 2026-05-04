"""T4 source-coverage upper bound: returns GT iff any source contains it.

This is explicitly an oracle: it consults persona-level ground truth at
predict time. Its purpose is to bound the headroom of any extraction-based
fusion method (the fraction of (persona, qid) pairs where the correct
answer is reachable from at least one source).

Unlike all other methods, `OracleExtraction` requires GT injection via
`attach_gt(persona_gts)` before prediction. The runner is responsible for
calling this once per evaluation pass with the test split's GT.
"""

from __future__ import annotations

from collections import Counter

from survey2agent.extraction.atoms import ExtractedAtom

from ._atom_adapter import atom_to_mu_q
from .base import SKIP_SENTINEL, Method, Prediction


class OracleExtraction(Method):
    """Returns GT label iff some source extracted it; otherwise plurality vote."""

    name = "OracleExtraction"
    requires_fit = False
    requires_calibration = False

    def __init__(self, *, skip_on_miss: bool = False) -> None:
        self.skip_on_miss = skip_on_miss
        self._gts: dict[str, dict[str, str]] = {}

    def attach_gt(self, persona_gts: dict[str, dict[str, str]]) -> None:
        """Bind `{persona_id: {qid: gt_label}}` for predict-time lookup."""
        self._gts = {p: dict(g) for p, g in persona_gts.items()}

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        gt = self._gts.get(atom.persona, {}).get(qid)
        mu_q = atom_to_mu_q(atom, qid)
        available = [v for v in mu_q.values() if v is not None]

        if gt is not None and gt in available:
            return Prediction(answer=gt, would_skip=False)

        # Coverage gap: GT missing from sources (or no attached GT).
        if self.skip_on_miss:
            return Prediction(answer=SKIP_SENTINEL, would_skip=True)
        if available:
            top = Counter(available).most_common(1)[0][0]
            return Prediction(answer=top, would_skip=False)
        return Prediction(answer=SKIP_SENTINEL, would_skip=True)
