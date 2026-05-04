"""Pin SHA256 hashes of the bundled prompt files.

These prompts are part of the artifact so the live LLM dispatch path
(when implemented) is reproducible. A change to any prompt is a spec
change and must be done deliberately; this test guards against silent
edits.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

_PROMPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "survey2agent"
    / "methods"
    / "prompts"
)

EXPECTED_SHA256: dict[str, str] = {
    "direct.md": "971e96d0862c1a0bb5479f6c6520b11497955da9f79776ff15298f6cd26e279c",
    "schema_aware.md": "f463ab2fb8c23d7a8bf6bc0a49cc0fb0e3428e02d2e31ef776d5f2166aa2fdf0",
    "questions_schema_aware.md": "4740eaaa606036a83ac31ed065b96e911e8c5ad53c95f6a3fee413fc3e15af0e",
    "few_shot.md": "2fed795ee971cd6b1a7ed7183a2c9ff1c8851147eade27cfa2381a53a408774a",
}


@pytest.mark.parametrize("name,expected", list(EXPECTED_SHA256.items()))
def test_prompt_sha256_pinned(name: str, expected: str) -> None:
    path = _PROMPTS_DIR / name
    assert path.exists(), f"prompt file missing: {path}"
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, (
        f"prompt hash mismatch for {name}: expected {expected}, got {actual}. "
        "If this change is intentional, update EXPECTED_SHA256."
    )
