"""Anonymization audit for double-blind submission.

Scans the repository tree for patterns that may de-anonymize the authors.
Reports findings to stdout. Exits non-zero if any non-allowlisted hit is
found, so this can be wired into CI.

Usage:
    python scripts/anonymization_audit.py
    python scripts/anonymization_audit.py --root . --strict

Detections:
  1. Real-looking email addresses (anything not a known placeholder).
  2. GitHub URLs whose owner segment is not in the allow-list of
     placeholder names ("anonymous", "anon-*", "ed-2026-*", ...).
  3. ORCID iDs.
  4. Lab / institution / personal-name strings declared in
     `ANONYMIZATION_FORBIDDEN`. Edit that list when concrete names need
     to be checked for (kept empty in the public template; populate
     locally before submission and revert before pushing).

With ``--check-release-surface`` an extra pass scans for internal-jargon
and absolute-path leakage patterns (cp-v2 IDs, in-repo prototype paths,
dev-tooling names, host filesystem paths). This is intended to be run
against a freshly assembled 4open staging tree (see
``scripts/build_4open_dump.sh``) rather than against the development
checkout.

Pass conditions: zero findings outside the allow-list; otherwise exits
with code 1 and a numbered list of offending file:line snippets.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parent.parent

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
GITHUB_RE = re.compile(r"github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)")
ORCID_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")

# Allow-list of email addresses considered anonymized placeholders.
EMAIL_ALLOWLIST = {
    # Generic / official addresses commonly cited in docs.
    "info@example.com",
    "noreply@github.com",
    "support@github.com",
    "datasets@huggingface.co",
    "support@dataverse.harvard.edu",
    "kaggle-datasets@kaggle.com",
    "openmlhq@openml.org",
    "evaluationsdatasets@neurips.cc",
    "datasetsbenchmarks@neurips.cc",
}

# Allow-list of GitHub *owner* segments.  The repository-name segment can
# be anything, but the owner must be a placeholder.
GITHUB_OWNER_ALLOWLIST = {
    "anonymous",
    "anon-neuripsed26",
    "anon-ed-2026",
    "ed-2026-anon",
    "mlcommons",
    "neurips",
    "neurips-org",
    "Croissant",
    "huggingface",
    "datasets",
    "openml",
    "openai",
    "google",
    "google-research",
    "AnthropicResearch",
    "anthropic",
    "modelcards",
    "datacards",
    "tensorflow",
    "pytorch",
}

# Filenames / glob suffixes that are skipped wholesale.
SKIP_FILE_SUFFIXES = {
    ".pyc",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".so",
    ".dylib",
    ".whl",
}

SKIP_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    "node_modules",
    ".egg-info",
    "egg-info",
    ".venv",
    "venv",
    ".mypy_cache",
}

# Personal / institutional strings that MUST not appear.  The template
# ships empty; populate locally with concrete names you want flagged
# (for example real first/last names, lab names, university names).
ANONYMIZATION_FORBIDDEN: tuple[str, ...] = ()

# Release-surface leakage patterns.  These are not de-anonymising on
# their own but mark internal-jargon, absolute filesystem paths, and
# dev-only tooling identifiers that should never appear in the
# 4open-bound staging tree.  Each entry is a compiled regex.
RELEASE_SURFACE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cp-v2-id", re.compile(r"\bcp-v2-\d{4}\b")),
    ("pre-release-prototype-path", re.compile(r"\bprototype" r"V2/")),
    ("abs-users-path", re.compile(r"/Users/[A-Za-z0-9_.\-]+")),
    ("abs-home-path", re.compile(r"/home/[A-Za-z0-9_.\-]+")),
    ("abs-github-path", re.compile(r"Documents/" r"GitHub/")),
    ("co-authored-by", re.compile(r"Co-authored-by:", re.IGNORECASE)),
    # Internal release-tracking dialects (wave_5e, wave5g, phase2_2,
    # phase2_3, etc.) that are not de-anonymising on their own but
    # broadcast a project's internal cadence.
    ("wave-id", re.compile(r"\bwave[\s_-]?\d+[a-z]?\b", re.IGNORECASE)),
    ("phase-id", re.compile(r"\bphase[\s_-]?\d+[\s_-]?\d+\b", re.IGNORECASE)),
)

# Lines matching any of these substrings are exempted from the
# release-surface scan (used to silence the audit script's own regex
# literals and intentional documentation references).
RELEASE_SURFACE_LINE_ALLOWLIST: tuple[str, ...] = (
    "RELEASE_SURFACE_PATTERNS",
    "RELEASE_SURFACE_LINE_ALLOWLIST",
)


def iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIR_NAMES or part.endswith(".egg-info") for part in p.parts):
            continue
        if p.suffix.lower() in SKIP_FILE_SUFFIXES:
            continue
        # Skip the audit script itself to avoid reporting its own
        # documentation / regex literals.  Match by basename so that a
        # copy of this script inside a staging tree is also skipped.
        if p.name == Path(__file__).name:
            continue
        files.append(p)
    return files


def scan_text(path: Path, text: str) -> list[tuple[int, str, str]]:
    findings: list[tuple[int, str, str]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for m in EMAIL_RE.finditer(line):
            email = m.group(0)
            if email.lower() in EMAIL_ALLOWLIST:
                continue
            findings.append((line_no, "email", email))
        for m in GITHUB_RE.finditer(line):
            owner = m.group(1)
            repo_name = m.group(2)
            if owner.lower() in {x.lower() for x in GITHUB_OWNER_ALLOWLIST}:
                continue
            findings.append(
                (line_no, "github", f"{owner}/{repo_name}")
            )
        for m in ORCID_RE.finditer(line):
            findings.append((line_no, "orcid", m.group(0)))
        for needle in ANONYMIZATION_FORBIDDEN:
            if needle and needle in line:
                findings.append((line_no, "forbidden", needle))
    return findings


def scan_release_surface(path: Path, text: str) -> list[tuple[int, str, str]]:
    """Scan for release-surface leakage patterns.

    Independent of :func:`scan_text` so that the standard anon audit
    keeps its meaning (de-anonymisation only) while the release-mode
    pass focuses on internal-jargon / absolute-path leakage.
    """
    findings: list[tuple[int, str, str]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if any(skip in line for skip in RELEASE_SURFACE_LINE_ALLOWLIST):
            continue
        for kind, pattern in RELEASE_SURFACE_PATTERNS:
            for m in pattern.finditer(line):
                findings.append((line_no, kind, m.group(0)))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT_DEFAULT,
        help="Directory to scan (default: repository root).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero on any finding. Without this flag, findings "
            "are printed but the audit always exits 0."
        ),
    )
    parser.add_argument(
        "--check-release-surface",
        action="store_true",
        help=(
            "In addition to the standard anonymisation audit, scan for "
            "internal-jargon / absolute-path leakage patterns (cp-v2 IDs, "
            "in-repo prototype paths, dev-tooling names, host filesystem "
            "paths). Intended for running against a 4open staging tree."
        ),
    )
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.exists():
        print(f"ERROR: --root {root} does not exist", file=sys.stderr)
        return 2

    total_findings = 0
    files = iter_files(root)
    print(f"[anonymization-audit] scanning {len(files)} files under {root}")

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        findings = scan_text(path, text)
        if args.check_release_surface:
            findings = findings + scan_release_surface(path, text)
        if not findings:
            continue
        total_findings += len(findings)
        rel = path.relative_to(root)
        for line_no, kind, snippet in findings:
            print(f"  {rel}:{line_no}  [{kind}]  {snippet}")

    print(f"[anonymization-audit] total findings: {total_findings}")
    if total_findings == 0:
        print("[anonymization-audit] OK -- no de-anonymization risks detected.")
        return 0
    if args.strict:
        print("[anonymization-audit] FAIL (--strict): findings present.")
        return 1
    print("[anonymization-audit] WARN: findings present (--strict not set).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
