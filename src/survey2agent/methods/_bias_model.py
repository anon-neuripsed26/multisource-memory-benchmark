"""Shared bias model for BCF and ABF.

Forward BiasPredict: given a hypothesis (candidate answer label) and a source,
return what the source *would* report if that candidate were the latent truth.

Algorithm (matches legacy `v1.0 reference`):

* nominal answer_space, or zero per-source bias `b` → identity
* candidate sits outside `ordinal_encoding` (e.g. an edge option) → identity
* otherwise: shift the candidate's ordinal rank by `b`, clamp to ``[1, K]``
  where ``K = len(ordinal_encoding)``, and reverse-lookup the label.

Interface change vs. legacy: the new `question_spec.get_bias(qid)` returns a
``{source: int}`` dict (per QBD-2 it resolves topic-dependent bias against
the question's topic). We index it as ``get_bias(qid)[source]``.
"""

from __future__ import annotations

from survey2agent.extraction.question_spec import QUESTIONS, get_bias


def bias_predict(source: str, qid: str, candidate: str) -> str:
    """Return what `source` would report if `candidate` were the truth.

    Raises:
        KeyError: if `qid` is not a known question id, or if `source` is not
            in the canonical source list (delegated to ``get_bias``).
    """
    q = QUESTIONS[qid]
    b = get_bias(qid)[source]

    # Nominal answer space, or unbiased source → identity.
    if q["answer_space_type"] == "nominal" or b == 0:
        return candidate

    enc = q.get("ordinal_encoding")
    if enc is None or candidate not in enc:
        # Edge option (no_frequency_described, no_plans, ...) or missing
        # encoding → identity per QBD-3.
        return candidate

    K = len(enc)
    ord_v = enc[candidate]
    shifted = max(1, min(K, ord_v + b))

    # Reverse lookup: ordinal rank → label.
    inv = {pos: label for label, pos in enc.items()}
    return inv[shifted]
