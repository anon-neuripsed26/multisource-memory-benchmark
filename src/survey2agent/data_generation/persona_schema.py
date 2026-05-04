"""Dataclass definitions for persona records and their sub-structures.

Every generated persona is represented as a `Persona` dataclass which
serialises to / deserialises from the benchmark ``personas.json`` file.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


# ── Profile ──────────────────────────────────────────────────────────────────

@dataclass
class Profile:
    age: int
    gender: str
    occupation: str
    relationship: str
    city: str


# ── Stable traits ────────────────────────────────────────────────────────────

@dataclass
class StableTraits:
    dietary_restriction: str
    social_preference: str
    close_friends: int
    primary_exercise: str


# ── Domain-level behavioural params ──────────────────────────────────────────

@dataclass
class SleepParams:
    bedtime_mean: str
    bedtime_std_min: int
    duration_mean_h: float
    duration_std_h: float
    quality_mean: float
    quality_std: float
    trouble_falling_asleep_prob: float
    screen_before_bed_prob: float


@dataclass
class ExerciseParams:
    days_per_week: float
    preferred_days: list[str] | None
    type: str
    duration_mean_min: int
    duration_std_min: int
    skip_prob: float


@dataclass
class DietParams:
    meals_per_day: float
    home_cook_prob: float
    fast_food_prob: float
    coffee_cups_mean: float
    coffee_cups_std: float


@dataclass
class WorkParams:
    hours_mean: float
    hours_std: float
    overtime_prob: float
    weekend_work_prob: float
    stress_prob: float


@dataclass
class SocialParams:
    meetup_prob_per_day: float
    weekend_social_boost: float
    social_activities: list[str]


@dataclass
class MoodParams:
    overall_mean: float
    overall_std: float
    stress_prob: float
    energy_mean: float
    energy_std: float
    relaxation_prob: float
    relaxation_activities: list[str]


@dataclass
class BehavioralParams:
    sleep: SleepParams
    exercise: ExerciseParams
    diet: DietParams
    work: WorkParams
    social: SocialParams
    mood: MoodParams


# ── Temporal shift ───────────────────────────────────────────────────────────

@dataclass
class TemporalShift:
    shift_day: int
    event_description: str
    shifted_params: BehavioralParams
    long_memory_reflects: str = "pre_shift"


# ── Stated vs Revealed ───────────────────────────────────────────────────────

@dataclass
class SparsityInjection:
    # NOTE on `missing_field_prob`:
    # This field is sampled but NEVER consumed by source_projector.py (it was
    # historically wired to a knob that field-level dropout never implemented).
    # The published dataset (`$S2A_DATA_ROOT/benchmark/seeds/s20260321/` and
    # the 4-seed sibling datasets) was generated with this field present, and
    # removing it shifts the numpy RNG sequence for the stated_vs_revealed
    # track, breaking byte-level reproducibility of all stated personas.
    # We retain it here so that re-running the data generation reproduces the
    # exact dataset used in the paper. The field is sampled but not
    # consumed by the persona constructor; it is retained to preserve
    # byte-level dataset reproducibility.
    missing_field_prob: float
    missing_day_prob: float


@dataclass
class StatedProfile:
    sleep_description: str
    exercise_description: str
    diet_description: str
    work_description: str
    social_description: str
    mood_description: str


@dataclass
class StatedVsRevealed:
    stated_profile: StatedProfile
    sparsity_injection: SparsityInjection


# ── Semantic conflict profile ────────────────────────────────────────────────

@dataclass
class AttributionConflict:
    for_others_food_order_prob: float
    shared_order_prob: float
    team_coffee_run_prob: float
    beneficiary_bias: str


@dataclass
class ConditionalPreferenceConflict:
    exception_contexts: list[str]
    self_meat_baseline_prob: float
    self_meat_on_exception_prob: float
    takeout_baseline_prob: float
    takeout_on_stress_prob: float


@dataclass
class SummaryCompressionConflict:
    coarse_long_memory_summary: str
    latent_behavior_state: str
    compressed_away: list[str]
    profile_summary_text: str


@dataclass
class IntentActionSemanticMismatch:
    personal_looking_work_meal_prob: float
    errand_for_other_prob: float
    intent_proxy_strength_mean: float
    action_for_self_vs_other_bias: str


@dataclass
class DietSemanticPatterns:
    planned_plant_based_day_prob: float
    solo_meal_prob: float
    solo_routine_identity_bias: str


@dataclass
class SocialSemanticPatterns:
    obligation_social_prob: float
    supporting_other_social_prob: float
    low_energy_inperson_prob: float
    remote_social_prob: float


@dataclass
class WorkSemanticPatterns:
    afterhours_for_self_prob: float
    afterhours_team_need_prob: float
    support_work_prob: float


@dataclass
class ExerciseSemanticPatterns:
    incidental_activity_prob: float
    tracker_false_positive_prob: float
    intentional_exercise_identity_prob: float


@dataclass
class SemanticConflictProfile:
    attribution_conflict: AttributionConflict
    conditional_preference_conflict: ConditionalPreferenceConflict
    summary_compression_conflict: SummaryCompressionConflict
    intent_action_semantic_mismatch: IntentActionSemanticMismatch
    diet_semantic_patterns: DietSemanticPatterns
    social_semantic_patterns: SocialSemanticPatterns
    work_semantic_patterns: WorkSemanticPatterns
    exercise_semantic_patterns: ExerciseSemanticPatterns


# ── Top-level Persona ────────────────────────────────────────────────────────

@dataclass
class Persona:
    id: str
    name: str
    description: str
    difficulty_type: str  # stable | temporal_shift | stated_vs_revealed
    profile: Profile
    stable_traits: StableTraits
    behavioral_params: BehavioralParams
    semantic_conflict_profile: SemanticConflictProfile
    temporal_shift: TemporalShift | None = None
    stated_vs_revealed: StatedVsRevealed | None = None


# ── Serialisation helpers ────────────────────────────────────────────────────

def _to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass tree to plain dicts/lists."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_dict(v) for v in obj]
    return obj


def persona_to_dict(persona: Persona) -> dict[str, Any]:
    """Serialize a ``Persona`` to a JSON-compatible dict."""
    return _to_dict(persona)


def behavioral_params_from_dict(d: dict[str, Any]) -> BehavioralParams:
    """Reconstruct a ``BehavioralParams`` from a plain dict."""
    return BehavioralParams(
        sleep=SleepParams(**d["sleep"]),
        exercise=ExerciseParams(**d["exercise"]),
        diet=DietParams(**d["diet"]),
        work=WorkParams(**d["work"]),
        social=SocialParams(**d["social"]),
        mood=MoodParams(**d["mood"]),
    )
