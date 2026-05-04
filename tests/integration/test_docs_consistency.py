"""Docs/code consistency lint.

Walks the Tier B markdown docs and verifies that every command-block
reference to:

  - `python3 -m <module>`  → `<module>` is importable.
  - `make <target>`        → `<target>` is declared in Makefile.
  - `from X import Y`      → `Y` is an attribute of `X` (importable).

Goal: catch the kind of doc rot in the appendix tier
(fake `paper_artifacts.run_method_eval`, fake `EventLog`, fake
`m._all_qids_and_specs`, fake `make seed-*`).

The lint is intentionally conservative: anything that looks like a
placeholder (`<...>`, `${...}`, contains `your_`/`my_`/`sample`/`example`
prefix on the symbol) is skipped. False positives on a real symbol mean
the doc is broken.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = [
    REPO_ROOT / "EXTENDING.md",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "SUBMISSION_PROTOCOL.md",
    REPO_ROOT / "ANONYMIZATION.md",
    REPO_ROOT / "CODE_OF_CONDUCT.md",
    REPO_ROOT / "SECURITY.md",
    REPO_ROOT / "examples" / "README.md",
    REPO_ROOT / "examples" / "01_minimal_method" / "README.md",
    REPO_ROOT / "examples" / "02_custom_question" / "README.md",
    REPO_ROOT / "examples" / "03_programmatic_api" / "README.md",
    REPO_ROOT / "examples" / "04_sample_submission" / "README.md",
    REPO_ROOT / "examples" / "05_custom_stream" / "README.md",
]
MAKEFILE = REPO_ROOT / "Makefile"

PY_MODULE_RE = re.compile(r"python3?\s+-m\s+([a-zA-Z_][\w\.]+)")
MAKE_RE = re.compile(r"`make\s+([a-zA-Z_][\w-]*)`")
FROM_IMPORT_RE = re.compile(
    r"from\s+([a-zA-Z_][\w\.]+)\s+import\s+([A-Za-z_][\w]*)"
)
PLACEHOLDER_TOKENS = (
    "your_", "your", "my_", "sample", "example", "h1_", "fixture", "synth",
)


def _make_targets() -> set[str]:
    targets: set[str] = set()
    for line in MAKEFILE.read_text().splitlines():
        m = re.match(r"^([a-zA-Z_][\w-]*)\s*:", line)
        if m and not line.startswith("\t"):
            targets.add(m.group(1))
        # Pattern targets like seed-%
        m2 = re.match(r"^([a-zA-Z_][\w-]*-)%\s*:", line)
        if m2 and not line.startswith("\t"):
            targets.add(m2.group(1))
    # Phony declarations
    for line in MAKEFILE.read_text().splitlines():
        if line.startswith(".PHONY:"):
            for tok in line.split(":", 1)[1].split():
                targets.add(tok)
    return targets


def _is_placeholder(symbol: str) -> bool:
    s = symbol.lower()
    return any(tok in s for tok in PLACEHOLDER_TOKENS)


def _doc_lines(doc: Path) -> list[tuple[int, str]]:
    return list(enumerate(doc.read_text().splitlines(), 1))


@pytest.fixture(scope="module")
def make_targets() -> set[str]:
    return _make_targets()


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_doc_python_module_invocations_are_importable(doc: Path) -> None:
    """Every `python3 -m <module>` in this doc resolves to an importable module."""
    if not doc.exists():
        pytest.skip(f"{doc} missing")
    failures: list[str] = []
    for lineno, line in _doc_lines(doc):
        for module in PY_MODULE_RE.findall(line):
            if _is_placeholder(module):
                continue
            try:
                importlib.import_module(module)
            except Exception as exc:  # ImportError, AttributeError, etc.
                failures.append(f"{doc.name}:{lineno}: `{module}` not importable ({exc.__class__.__name__})")
    if failures:
        pytest.fail("\n".join(failures))


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_doc_make_targets_exist(doc: Path, make_targets: set[str]) -> None:
    """Every `make <target>` in this doc is declared in the Makefile."""
    if not doc.exists():
        pytest.skip(f"{doc} missing")
    # Pattern targets accept any concrete suffix (e.g. seed-s20260321 → seed-).
    pattern_prefixes = {t for t in make_targets if t.endswith("-")}

    failures: list[str] = []
    for lineno, line in _doc_lines(doc):
        for target in MAKE_RE.findall(line):
            if _is_placeholder(target):
                continue
            if target in make_targets:
                continue
            if any(target.startswith(p) for p in pattern_prefixes):
                continue
            failures.append(f"{doc.name}:{lineno}: `make {target}` not declared in Makefile")
    if failures:
        pytest.fail("\n".join(failures))


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_doc_from_imports_resolve(doc: Path) -> None:
    """Every `from X import Y` in this doc resolves: X importable and Y on it."""
    if not doc.exists():
        pytest.skip(f"{doc} missing")
    failures: list[str] = []
    for lineno, line in _doc_lines(doc):
        for module, symbol in FROM_IMPORT_RE.findall(line):
            if _is_placeholder(module) or _is_placeholder(symbol):
                continue
            try:
                mod = importlib.import_module(module)
            except Exception as exc:
                failures.append(
                    f"{doc.name}:{lineno}: `from {module} import {symbol}` "
                    f"-- module {module} not importable ({exc.__class__.__name__})"
                )
                continue
            if not hasattr(mod, symbol):
                failures.append(
                    f"{doc.name}:{lineno}: `from {module} import {symbol}` "
                    f"-- {symbol} not found on {module}"
                )
    if failures:
        pytest.fail("\n".join(failures))
