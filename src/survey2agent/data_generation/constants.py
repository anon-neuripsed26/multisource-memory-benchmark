"""Constants for persona generation: names, cities, occupations, parameter ranges.

All domain-specific enumerations and value ranges live here so that generator
modules stay free of magic literals.
"""

from __future__ import annotations

# ── Calendar ──────────────────────────────────────────────────────────────────

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# ── Demographics ──────────────────────────────────────────────────────────────

FIRST_NAMES: tuple[str, ...] = (
    "Avery", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Cameron", "Parker", "Reese",
    "Alex", "Maya", "Logan", "Skyler", "Harper", "Elliot", "Sage", "Rowan", "Blake", "Emerson",
    "Drew", "Jules", "Noah", "Leah", "Miles", "Nina", "Evan", "Priya", "Owen", "Lena",
    "Caleb", "Sofia", "Julian", "Naomi", "Isaac", "Mila", "Theo", "Aisha", "Leo", "Clara",
    "Felix", "Tessa", "Micah", "Zoe", "Adrian", "Ivy", "Simon", "Elena", "Omar", "Hazel",
    "Sam", "Ruby", "Aria", "Jonah", "Luca", "Maeve", "Nora", "Ezra", "Maren", "Kai",
    "Vera", "Liam", "Dahlia", "Milo", "Jasper", "Cora", "Rohan", "Amelia", "Kira", "Sebastian",
)

LAST_NAMES: tuple[str, ...] = (
    "Adams", "Bennett", "Carter", "Diaz", "Ellis", "Foster", "Garcia", "Hayes", "Iqbal", "Jenkins",
    "Kim", "Lopez", "Morgan", "Nguyen", "Owens", "Patel", "Quintero", "Ramirez", "Singh", "Turner",
    "Underwood", "Vasquez", "Walker", "Xu", "Young", "Zimmerman", "Brooks", "Campbell", "Davis", "Edwards",
    "Flores", "Griffin", "Howard", "Ivanov", "Jackson", "Khan", "Larson", "Mitchell", "Nakamura", "Ortiz",
)

GENDERS: tuple[str, ...] = ("female", "male", "non-binary")

OCCUPATIONS: tuple[str, ...] = (
    "software engineer", "teacher", "marketing manager", "product designer",
    "nurse", "financial analyst", "graduate student", "freelance writer",
    "architect", "physical therapist", "project manager", "chef",
    "accountant", "graphic designer", "data analyst", "sales manager",
    "research assistant", "operations specialist", "customer success manager", "UX researcher",
)

# Occupations considered "work-heavy" for semantic conflict profiling
WORK_HEAVY_OCCUPATIONS: frozenset[str] = frozenset({
    "software engineer", "product designer", "project manager",
    "operations specialist", "marketing manager", "financial analyst",
})

CITIES: tuple[str, ...] = (
    "Austin, TX", "Chicago, IL", "Portland, OR", "Seattle, WA", "Boston, MA",
    "Denver, CO", "Atlanta, GA", "San Diego, CA", "Minneapolis, MN", "Philadelphia, PA",
    "Nashville, TN", "Charlotte, NC", "Phoenix, AZ", "Madison, WI", "Brooklyn, NY",
    "Columbus, OH", "Ann Arbor, MI", "Raleigh, NC", "Pittsburgh, PA", "Salt Lake City, UT",
)

RELATIONSHIPS: tuple[str, ...] = ("single", "in a relationship", "married")

AGE_RANGE: tuple[int, int] = (24, 46)

# ── Behavioral: Dietary ──────────────────────────────────────────────────────

DIETARY_RESTRICTIONS: tuple[str, ...] = ("none", "vegetarian", "vegan", "low-carb", "gluten-free")
DIETARY_WEIGHTS: tuple[float, ...] = (0.55, 0.15, 0.10, 0.10, 0.10)

# ── Behavioral: Social ───────────────────────────────────────────────────────

SOCIAL_PREFERENCES: tuple[str, ...] = ("in person", "online", "no preference")
SOCIAL_PREF_WEIGHTS: tuple[float, ...] = (0.60, 0.15, 0.25)

SOCIAL_ACTIVITY_POOL: tuple[str, ...] = (
    "dinner with friends", "coffee with a colleague", "family brunch",
    "book club", "board game night", "pickup basketball",
    "bar trivia", "group hike", "birthday dinner",
    "video call with friends", "coworking session", "art gallery opening",
    "live music night", "church gathering", "neighborhood walk",
)

# ── Behavioral: Exercise ─────────────────────────────────────────────────────

EXERCISE_TYPES: tuple[str, ...] = (
    "running", "cycling", "yoga", "pilates", "strength training",
    "swimming", "walking", "dance", "hiking", "boxing",
)

# ── Behavioral: Relaxation ───────────────────────────────────────────────────

RELAXATION_ACTIVITIES: tuple[str, ...] = (
    "reading", "gaming", "watching TV", "meditation", "journaling",
    "podcasts", "stretching", "cooking", "music", "sketching",
    "gardening", "documentaries",
)

# ── Temporal Shift Scenarios (8 total) ────────────────────────────────────────

SHIFT_SCENARIOS: tuple[dict[str, object], ...] = (
    {
        "id": "promotion_crunch",
        "event_description": "Got promoted into a deadline-heavy leadership role",
        "sleep_delta": -1.5, "quality_delta": -1.5,
        "exercise_multiplier": 0.4, "diet_fast_food_boost": 0.35,
        "coffee_delta": 1.5, "work_hours_delta": 2.5,
        "overtime_boost": 0.45, "weekend_work_boost": 0.35,
        "social_multiplier": 0.35, "mood_delta": -1.0, "stress_boost": 0.35,
    },
    {
        "id": "family_caregiving",
        "event_description": "Started helping with intensive family caregiving responsibilities",
        "sleep_delta": -1.2, "quality_delta": -1.2,
        "exercise_multiplier": 0.5, "diet_fast_food_boost": 0.20,
        "coffee_delta": 1.0, "work_hours_delta": 1.0,
        "overtime_boost": 0.20, "weekend_work_boost": 0.10,
        "social_multiplier": 0.45, "mood_delta": -0.8, "stress_boost": 0.25,
    },
    {
        "id": "wellness_reset",
        "event_description": "Committed to a structured wellness reset after a stressful quarter",
        "sleep_delta": 0.8, "quality_delta": 0.8,
        "exercise_multiplier": 1.7, "diet_fast_food_boost": -0.15,
        "coffee_delta": -0.5, "work_hours_delta": -0.5,
        "overtime_boost": -0.10, "weekend_work_boost": -0.10,
        "social_multiplier": 0.9, "mood_delta": 0.8, "stress_boost": -0.20,
    },
    {
        "id": "launch_crunch",
        "event_description": "Took on a compressed launch cycle for a major client project",
        "sleep_delta": -1.0, "quality_delta": -1.0,
        "exercise_multiplier": 0.5, "diet_fast_food_boost": 0.25,
        "coffee_delta": 1.0, "work_hours_delta": 3.0,
        "overtime_boost": 0.50, "weekend_work_boost": 0.40,
        "social_multiplier": 0.3, "mood_delta": -1.2, "stress_boost": 0.40,
    },
    {
        "id": "new_relationship",
        "event_description": "Started a new romantic relationship and social life expanded",
        "sleep_delta": -0.5, "quality_delta": 0.3,
        "exercise_multiplier": 0.7, "diet_fast_food_boost": 0.15,
        "coffee_delta": 0.3, "work_hours_delta": -0.5,
        "overtime_boost": -0.05, "weekend_work_boost": -0.05,
        "social_multiplier": 1.6, "mood_delta": 0.8, "stress_boost": -0.10,
    },
    {
        "id": "health_scare",
        "event_description": "Received a health scare and overhauled daily habits",
        "sleep_delta": 1.0, "quality_delta": 0.5,
        "exercise_multiplier": 1.5, "diet_fast_food_boost": -0.25,
        "coffee_delta": -1.0, "work_hours_delta": -1.0,
        "overtime_boost": -0.10, "weekend_work_boost": -0.10,
        "social_multiplier": 0.8, "mood_delta": -0.5, "stress_boost": 0.15,
    },
    {
        "id": "relocation",
        "event_description": "Relocated to a new city for a partner's job",
        "sleep_delta": -0.6, "quality_delta": -0.5,
        "exercise_multiplier": 0.6, "diet_fast_food_boost": 0.20,
        "coffee_delta": 0.5, "work_hours_delta": 0.5,
        "overtime_boost": 0.10, "weekend_work_boost": 0.05,
        "social_multiplier": 0.4, "mood_delta": -0.6, "stress_boost": 0.20,
    },
    {
        "id": "burnout_recovery",
        "event_description": "Hit burnout wall and drastically reduced commitments",
        "sleep_delta": 1.2, "quality_delta": 0.6,
        "exercise_multiplier": 0.8, "diet_fast_food_boost": 0.10,
        "coffee_delta": -0.3, "work_hours_delta": -2.0,
        "overtime_boost": -0.15, "weekend_work_boost": -0.10,
        "social_multiplier": 0.5, "mood_delta": 0.3, "stress_boost": -0.15,
    },
)

# ── Diversity constraints ────────────────────────────────────────────────────

TOTAL_PERSONAS = 480
PERSONAS_PER_DIFFICULTY = 160
MAX_CONCENTRATION_FRAC = 0.10  # <=10% of total per city/occupation

# ── Simulation window ────────────────────────────────────────────────────────

NUM_DAYS = 30
START_DATE = "2026-01-03"
END_DATE = "2026-02-01"
SURVEY_REFERENCE_DATE = "2026-02-01"

# ── Difficulty types ──────────────────────────────────────────────────────────

DIFFICULTY_TYPES: tuple[str, ...] = ("stable", "temporal_shift", "stated_vs_revealed")

# ── Semantic conflict families ────────────────────────────────────────────────

SEMANTIC_CONFLICT_FAMILIES: tuple[str, ...] = (
    "attribution_conflict",
    "conditional_preference_conflict",
    "summary_compression_conflict",
    "intent_action_semantic_mismatch",
    "diet_semantic_patterns",
    "social_semantic_patterns",
    "work_semantic_patterns",
    "exercise_semantic_patterns",
)
