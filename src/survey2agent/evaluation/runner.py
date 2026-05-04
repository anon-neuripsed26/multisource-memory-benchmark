"""Method runner: drive a method through fit → calibrate → predict.

A ``Method`` consumes per-persona method-facing records for fit and
calibrate (the schema declared in
:data:`survey2agent.methods.base.MethodTrainingRecord`),
and emits a :class:`Prediction` per ``(persona, qid)`` for evaluation.

This runner accepts the per-question :class:`TrainingRecord` from
:mod:`.data_loaders` and internally regroups it per-persona for the
``fit`` / ``calibrate`` calls.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence

from survey2agent.methods.base import Method, MethodTrainingRecord, Prediction
from survey2agent.selective.protocol import (
    EvaluationOutcome,
    classify_prediction,
)

from .data_loaders import TrainingRecord

__all__ = ["EvaluationResult", "run_method"]


@dataclass(frozen=True)
class EvaluationResult:
    """One ``(method, persona, qid)`` outcome with prediction + label.

    ``reasoning_type`` (e.g. ``"arbitration"``) and ``topic`` (e.g.
    ``"sleep"``) come from ``configs/questions.yaml``; ``difficulty_class``
    (one of ``"stable"`` / ``"temporal_shift"`` / ``"stated_vs_revealed"``)
    comes from the persona spec. Populated by :func:`run_method` from the
    incoming ``TrainingRecord`` so downstream per-type / per-topic /
    per-difficulty breakdowns require no re-loading of upstream specs.
    """

    method_name: str
    persona_id: str
    qid: str
    prediction: Prediction
    label: str
    outcome: EvaluationOutcome
    reasoning_type: str
    topic: str
    difficulty_class: str


def _to_method_records(
    records: Sequence[TrainingRecord],
) -> list[MethodTrainingRecord]:
    """Group per-question records into per-persona method-facing records.

    Preserves first-seen persona order. Each persona's GT dict contains
    only the qids that appear in ``records`` for that persona.
    """
    grouped: "OrderedDict[str, tuple]" = OrderedDict()
    for r in records:
        pid = r.atom.persona
        if pid not in grouped:
            grouped[pid] = (r.atom, {}, r.difficulty_class)
        else:
            _atom, _gt, difficulty_class = grouped[pid]
            if difficulty_class != r.difficulty_class:
                raise ValueError(
                    f"inconsistent difficulty_class for persona {pid!r}: "
                    f"{difficulty_class!r} vs {r.difficulty_class!r}"
                )
        grouped[pid][1][r.qid] = r.label
    return [
        MethodTrainingRecord(atom=atom, gt=dict(gt), difficulty_class=difficulty_class)
        for atom, gt, difficulty_class in grouped.values()
    ]


def run_method(
    method: Method,
    test_records: Sequence[TrainingRecord],
    *,
    train_records: Sequence[TrainingRecord] | None = None,
    cal_records: Sequence[TrainingRecord] | None = None,
) -> list[EvaluationResult]:
    """Execute ``fit`` (if required) → ``calibrate`` (if required) → predict.

    Order of ``EvaluationResult``s mirrors ``test_records``.
    """
    if method.requires_fit:
        if train_records is None:
            raise ValueError(
                f"method {method.name!r} requires_fit=True but train_records is None"
            )
        method.fit(_to_method_records(train_records))

    if method.requires_calibration:
        if cal_records is None:
            raise ValueError(
                f"method {method.name!r} requires_calibration=True but cal_records is None"
            )
        method.calibrate(_to_method_records(cal_records))

    out: list[EvaluationResult] = []
    for r in test_records:
        pred = method.predict_one(r.atom, r.qid)
        outcome = classify_prediction(pred, r.label)
        out.append(
            EvaluationResult(
                method_name=method.name,
                persona_id=r.atom.persona,
                qid=r.qid,
                prediction=pred,
                label=r.label,
                outcome=outcome,
                reasoning_type=r.reasoning_type,
                topic=r.topic,
                difficulty_class=r.difficulty_class,
            )
        )
    return out
