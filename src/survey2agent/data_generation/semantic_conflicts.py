"""Semantic conflict profile sampling (8 conflict families).

Each persona receives a ``SemanticConflictProfile`` controlling how sources
will disagree about that persona's behaviour.  Probability ranges are
difficulty-conditioned: *stated_vs_revealed* personas get wider conflict
margins than *stable* ones.
"""

from __future__ import annotations

from numpy.random import Generator

from .constants import WORK_HEAVY_OCCUPATIONS
from .persona_schema import (
    AttributionConflict,
    ConditionalPreferenceConflict,
    DietSemanticPatterns,
    ExerciseSemanticPatterns,
    IntentActionSemanticMismatch,
    SemanticConflictProfile,
    SocialSemanticPatterns,
    SummaryCompressionConflict,
    WorkSemanticPatterns,
)


# ── Summary-text look-up ─────────────────────────────────────────────────────

_PROFILE_SUMMARY_TEXT: dict[str, str] = {
    "vegan": "tries to eat vegan most of the time",
    "trying_to_be_vegan": "is trying to be vegan",
    "vegetarian": "usually keeps a vegetarian routine",
    "mostly_plant_based": "mostly eats plant-based meals",
    "health_conscious_but_flexible": "prefers healthy meals but stays flexible",
    "mixed_diet": "does not follow a strict diet label",
}


# ── Public entry point ───────────────────────────────────────────────────────

def sample_semantic_conflict_profile(
    *,
    difficulty_type: str,
    dietary_restriction: str,
    occupation: str,
    rng: Generator,
) -> SemanticConflictProfile:
    """Build a complete 8-family semantic conflict profile for one persona."""

    work_heavy = occupation in WORK_HEAVY_OCCUPATIONS

    # ── 1. Summary compression: coarse diet label ────────────────────────
    if dietary_restriction == "vegan":
        labels = ["vegan", "trying_to_be_vegan", "mostly_plant_based"]
        weights = [0.45, 0.35, 0.20]
    elif dietary_restriction == "vegetarian":
        labels = ["vegetarian", "trying_to_be_vegan", "mostly_plant_based"]
        weights = [0.50, 0.20, 0.30]
    else:
        labels = ["mostly_plant_based", "mixed_diet", "health_conscious_but_flexible"]
        weights = [0.25, 0.45, 0.30]

    w = [x / sum(weights) for x in weights]  # normalise for numpy
    coarse_summary: str = labels[int(rng.choice(len(labels), p=w))]

    # ── 2. Exception contexts ────────────────────────────────────────────
    exception_contexts: list[str] = []
    if rng.random() < 0.72:
        exception_contexts.append("social")
    if rng.random() < 0.46:
        exception_contexts.append("travel")
    if rng.random() < (0.58 if work_heavy else 0.36):
        exception_contexts.append("work_stress")
    if rng.random() < 0.34:
        exception_contexts.append("weekend")
    if not exception_contexts:
        options = ["social", "travel", "work_stress", "weekend"]
        exception_contexts = [options[int(rng.integers(len(options)))]]

    # ── 3. Difficulty-conditioned probability ranges ──────────────────────
    if difficulty_type == "stated_vs_revealed":
        self_exception_meat = _u(rng, 0.35, 0.62)
        self_base_meat = _u(rng, 0.12, 0.28)
        for_others_prob = _u(rng, 0.30, 0.58)
        work_proxy_prob = _u(rng, 0.26, 0.52)
    elif difficulty_type == "temporal_shift":
        self_exception_meat = _u(rng, 0.22, 0.48)
        self_base_meat = _u(rng, 0.08, 0.20)
        for_others_prob = _u(rng, 0.22, 0.45)
        work_proxy_prob = _u(rng, 0.22, 0.44)
    else:  # stable
        self_exception_meat = _u(rng, 0.12, 0.34)
        self_base_meat = _u(rng, 0.02, 0.14)
        for_others_prob = _u(rng, 0.12, 0.32)
        work_proxy_prob = _u(rng, 0.10, 0.28)

    # ── 4. Derived fields ────────────────────────────────────────────────
    plant_based_labels = {"vegan", "vegetarian", "trying_to_be_vegan", "mostly_plant_based"}

    latent_state = (
        "strict_vegan"
        if coarse_summary == "vegan" and self_base_meat <= 0.08
        else "plant_based_with_contextual_exceptions"
        if coarse_summary in plant_based_labels
        else "flexible_mixed_diet"
    )

    compressed_away = ["for_others_orders", "shared_orders", "contextual_exceptions"]
    if "work_stress" in exception_contexts:
        compressed_away.append("stress_takeout")
    if "travel" in exception_contexts:
        compressed_away.append("travel_relaxation")

    solo_bias = (
        "strict_routine"
        if coarse_summary in {"vegan", "vegetarian"} and self_base_meat <= 0.14
        else "routine_with_shared_exceptions"
        if coarse_summary in {"trying_to_be_vegan", "mostly_plant_based"}
        else "flexible"
    )

    beneficiary_options = ["partner", "friend", "coworker", "family"]
    self_other_options = ["self", "other", "mixed"]

    return SemanticConflictProfile(
        attribution_conflict=AttributionConflict(
            for_others_food_order_prob=for_others_prob,
            shared_order_prob=_u(rng, 0.14, 0.36),
            team_coffee_run_prob=_u(rng, 0.08, 0.24),
            beneficiary_bias=beneficiary_options[int(rng.integers(len(beneficiary_options)))],
        ),
        conditional_preference_conflict=ConditionalPreferenceConflict(
            exception_contexts=sorted(exception_contexts),
            self_meat_baseline_prob=self_base_meat,
            self_meat_on_exception_prob=self_exception_meat,
            takeout_baseline_prob=_u(rng, 0.08, 0.22),
            takeout_on_stress_prob=_u(rng, 0.32, 0.62),
        ),
        summary_compression_conflict=SummaryCompressionConflict(
            coarse_long_memory_summary=coarse_summary,
            latent_behavior_state=latent_state,
            compressed_away=sorted(set(compressed_away)),
            profile_summary_text=_PROFILE_SUMMARY_TEXT[coarse_summary],
        ),
        intent_action_semantic_mismatch=IntentActionSemanticMismatch(
            personal_looking_work_meal_prob=work_proxy_prob,
            errand_for_other_prob=_u(rng, 0.14, 0.30),
            intent_proxy_strength_mean=_u(rng, 0.42, 0.78),
            action_for_self_vs_other_bias=self_other_options[int(rng.integers(len(self_other_options)))],
        ),
        diet_semantic_patterns=DietSemanticPatterns(
            planned_plant_based_day_prob=(
                _u(rng, 0.52, 0.86) if coarse_summary in plant_based_labels
                else _u(rng, 0.18, 0.42)
            ),
            solo_meal_prob=_u(rng, 0.42, 0.78),
            solo_routine_identity_bias=solo_bias,
        ),
        social_semantic_patterns=SocialSemanticPatterns(
            obligation_social_prob=_u(rng, 0.05, 0.90),
            supporting_other_social_prob=_u(rng, 0.14, 0.38),
            low_energy_inperson_prob=_u(rng, 0.08, 0.24),
            remote_social_prob=_u(rng, 0.18, 0.36),
        ),
        work_semantic_patterns=WorkSemanticPatterns(
            afterhours_for_self_prob=_u(rng, 0.12, 0.34),
            afterhours_team_need_prob=_u(rng, 0.20, 0.48),
            support_work_prob=_u(rng, 0.18, 0.42),
        ),
        exercise_semantic_patterns=ExerciseSemanticPatterns(
            incidental_activity_prob=_u(rng, 0.14, 0.36),
            tracker_false_positive_prob=_u(rng, 0.18, 0.42),
            intentional_exercise_identity_prob=_u(rng, 0.48, 0.82),
        ),
    )


# ── Private helpers ──────────────────────────────────────────────────────────

def _u(rng: Generator, lo: float, hi: float) -> float:
    """Draw a single uniform float rounded to 2 d.p."""
    return round(float(rng.uniform(lo, hi)), 2)
