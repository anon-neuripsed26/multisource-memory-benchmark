"""Upload the full reproducibility bundle to a Hugging Face dataset repository.

Workflow (double-blind submission):

  1. Create a throwaway HF account (the one in use for this submission
     is ``anon-neuripsed26``). Verify the email but DO NOT add personal
     info, avatars, or organization links. Generate a write-scoped
     access token at ``https://huggingface.co/settings/tokens``.
  2. Create the dataset repository in the web UI (visibility: public,
     license cc-by-4.0). The repo for this submission is
     ``anon-neuripsed26/multisource-memory-benchmark``.
  3. Run::

         export HF_TOKEN=hf_xxxxx
         python3 scripts/upload_to_hf.py \\
             --repo-id anon-neuripsed26/multisource-memory-benchmark \\
             --data-root data

  4. After upload, set ``S2A_HF_REPO=<your-repo-id>`` so
     ``data/fetch_benchmark.py`` and ``make fetch`` pull from the new
     location during reviewer reproduction.

What this script does NOT do:

  * It does not anonymize the dataset itself. Run
    ``scripts/anonymization_audit.py --root data --strict``
    BEFORE uploading.
  * It does not commit. The dataset is a separate artifact from the
    code repository.

Repository layout produced on HF (mirrors local ``$S2A_DATA_ROOT/``):

  ``benchmark/``        — raw benchmark + per-method results (~370 MB)
  ``extracted_atoms/``  — frozen LLM-extracted atoms used by paper      (~2 MB)
  ``method_outputs/``   — frozen per-method output JSONs used by paper  (~30 MB)
  ``README.md``         — HF dataset card (YAML frontmatter + description)
  ``DATA_LICENSE``      — full text of CC-BY-4.0
  ``DATASHEET.md``      — datasheet-for-datasets answers
  ``CITATION.cff``      — anonymized citation entry
  ``CROISSANT_RAI.json`` — completed Croissant metadata with NeurIPS RAI fields

  Hugging Face generates the core Croissant metadata after upload. The
  completed ``CROISSANT_RAI.json`` is built from that endpoint plus the
  NeurIPS-required Responsible AI fields and is uploaded to OpenReview.

Files SKIPPED (hard-coded):

  * ``__pycache__/``, ``.DS_Store``, ``*.pyc``, ``.git/``, ``.pytest_cache/``
  * ``sample/`` (kept in the code repo only; HF hosts the full data)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Subdirectories under --data-root to upload, in order. Each maps to the
# same name at the HF repo root.
DATA_SUBDIRS: tuple[str, ...] = (
    "benchmark",
    "extracted_atoms",
    "method_outputs",
)

# Top-level metadata files to upload as-is to the HF repo root.
# Source resolution: try ``<data-root>/<file>`` first, then fall back to
# ``<data-root>.parent/<file>``. This lets canonical copies of
# ``DATASHEET.md`` lives at the code repo root
# while still being uploaded to HF.
#
# NOTE: no local placeholder ``CROISSANT.json`` is uploaded. The real
# completed Croissant+RAI file is ``CROISSANT_RAI.json``.
METADATA_FILES: tuple[str, ...] = (
    "README.md",
    "DATA_LICENSE",
    "DATASHEET.md",
    "CITATION.cff",
    "CROISSANT_RAI.json",
)


def _resolve_metadata(name: str, data_root: Path) -> Path | None:
    """Return the source path for a metadata file, or None if missing."""
    candidates = [data_root / name, data_root.parent / name]
    for c in candidates:
        if c.is_file():
            return c
    return None

# Patterns ignored under every subdirectory upload.
IGNORE_PATTERNS: tuple[str, ...] = (
    "**/__pycache__/**",
    "**/.DS_Store",
    "**/*.pyc",
    "**/.git/**",
    "**/.pytest_cache/**",
)


def _enumerate_subdir(root: Path) -> list[Path]:
    """Recursively enumerate uploadable files under ``root`` (excluding ignores).

    The exclusion logic mirrors :data:`IGNORE_PATTERNS` so that the
    dry-run output matches what ``upload_folder`` will actually push.
    """
    skip_parts = {"__pycache__", ".git", ".pytest_cache"}
    skip_names = {".DS_Store"}
    skip_suffixes = {".pyc"}
    files: list[Path] = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        parts = set(f.relative_to(root).parts)
        if parts & skip_parts:
            continue
        if f.name in skip_names:
            continue
        if f.suffix in skip_suffixes:
            continue
        files.append(f)
    return files


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--repo-id",
        required=True,
        help="HF dataset repo id (e.g. anon-neuripsed26/multisource-memory-benchmark)",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Local data root containing benchmark/, extracted_atoms/, method_outputs/, and metadata files (default: data)",
    )
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF write token (or set HF_TOKEN env var)")
    p.add_argument("--commit-message", default="Upload survey2agent reproducibility bundle (double-blind)")
    p.add_argument("--dry-run", action="store_true", help="List files that would be uploaded; do not push")
    args = p.parse_args()

    if not args.token and not args.dry_run:
        print("ERROR: HF token required. Set HF_TOKEN env var or pass --token.", file=sys.stderr)
        return 2

    data_root: Path = args.data_root.resolve()
    if not data_root.exists():
        print(f"ERROR: data root not found: {data_root}", file=sys.stderr)
        return 2

    # Validate metadata files exist (under data-root or one level up).
    meta_resolved: dict[str, Path] = {}
    missing_meta: list[str] = []
    for f in METADATA_FILES:
        src = _resolve_metadata(f, data_root)
        if src is None:
            missing_meta.append(f)
        else:
            meta_resolved[f] = src
    if missing_meta:
        print(
            "ERROR: missing required metadata files (looked under "
            f"{data_root} and {data_root.parent}): {', '.join(missing_meta)}",
            file=sys.stderr,
        )
        print(
            "       Generate them before uploading.",
            file=sys.stderr,
        )
        return 2

    # All three data subdirectories are MANDATORY for the full
    # reproducibility bundle. Refusing to upload a partial bundle
    # prevents reviewers from getting a dataset that cannot reproduce
    # the paper's tables.
    missing_subdirs = [s for s in DATA_SUBDIRS if not (data_root / s).is_dir()]
    if missing_subdirs:
        print(
            f"ERROR: missing required data subdirectories under "
            f"{data_root}: {', '.join(missing_subdirs)}. The HF release "
            f"is the FULL reproducibility bundle and requires all of "
            f"{DATA_SUBDIRS}.",
            file=sys.stderr,
        )
        return 2
    present_subdirs = list(DATA_SUBDIRS)

    # Enumerate everything for dry-run accounting.
    plan: list[tuple[str, Path, list[Path]]] = []  # (subdir_name, root, files)
    grand_total_bytes = 0
    grand_total_files = 0
    for sub in present_subdirs:
        sub_root = data_root / sub
        files = _enumerate_subdir(sub_root)
        sub_bytes = sum(f.stat().st_size for f in files)
        plan.append((sub, sub_root, files))
        grand_total_bytes += sub_bytes
        grand_total_files += len(files)
        print(
            f"  {sub:<20s} {len(files):>6d} files  {sub_bytes / 1e6:>8.1f} MB"
            f"  ({sub_root})"
        )

    meta_bytes = sum(p.stat().st_size for p in meta_resolved.values())
    print(
        f"  {'(metadata files)':<20s} {len(METADATA_FILES):>6d} files  "
        f"{meta_bytes / 1e3:>8.1f} KB  (README/LICENSE/DATASHEET/CITATION)"
    )
    print(
        f"  {'TOTAL':<20s} {grand_total_files + len(METADATA_FILES):>6d} files  "
        f"{(grand_total_bytes + meta_bytes) / 1e6:>8.1f} MB"
    )

    if args.dry_run:
        print()
        print("=== Resolved metadata sources ===")
        for name, src in meta_resolved.items():
            print(f"  {name:<18s} <- {src}")
        print()
        print("=== Sample of files per subdir (first 5) ===")
        for sub, root, files in plan:
            print(f"  [{sub}/]")
            for f in sorted(files)[:5]:
                print(f"    {f.relative_to(root)}")
            if len(files) > 5:
                print(f"    ... ({len(files) - 5} more)")
        print()
        print("(dry-run; nothing uploaded)")
        return 0

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        return 2

    api = HfApi(token=args.token)
    print(f"\nEnsuring dataset repo exists: {args.repo_id}")
    create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True, token=args.token)

    # Upload metadata files first (so the repo is browsable before the
    # large folders finish).
    for meta in METADATA_FILES:
        meta_path = meta_resolved[meta]
        print(f"Uploading metadata: {meta} ({meta_path.stat().st_size / 1e3:.1f} KB)  [from {meta_path}]")
        api.upload_file(
            path_or_fileobj=str(meta_path),
            path_in_repo=meta,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"{args.commit_message} ({meta})",
        )

    # Upload each data subdirectory under its own commit so that an
    # interrupted upload is retryable folder-by-folder. Use
    # upload_large_folder for subdirs whose file count exceeds HF's
    # single-commit threshold (~10k); it batches into multiple commits
    # automatically and supports resume on interruption.
    LARGE_FOLDER_THRESHOLD = 5000
    for sub, sub_root, files in plan:
        n_files = len(files)
        size_mb = sum(f.stat().st_size for f in files) / 1e6
        print(
            f"Uploading {sub}/ -> {sub}/  ({n_files} files, "
            f"{size_mb:.1f} MB; this can take several minutes)..."
        )
        if n_files >= LARGE_FOLDER_THRESHOLD:
            print(
                f"  [{n_files} files >= {LARGE_FOLDER_THRESHOLD}; using upload_large_folder "
                f"(multi-commit, resumable)]"
            )
            # upload_large_folder does not accept path_in_repo. Instead
            # we point folder_path at data_root (which already mirrors
            # the desired repo layout) and use allow_patterns to scope
            # to this subdir only. Files land at repo root preserving
            # the {sub}/... prefix.
            api.upload_large_folder(
                repo_id=args.repo_id,
                repo_type="dataset",
                folder_path=str(data_root),
                allow_patterns=[f"{sub}/**"],
                ignore_patterns=list(IGNORE_PATTERNS),
            )
        else:
            api.upload_folder(
                folder_path=str(sub_root),
                path_in_repo=sub,
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message=f"{args.commit_message} ({sub})",
                ignore_patterns=list(IGNORE_PATTERNS),
            )

    print(f"\nDone. Dataset is now at: https://huggingface.co/datasets/{args.repo_id}")
    print()
    print("Next steps:")
    print(f"  1. export S2A_HF_REPO={args.repo_id}")
    print(f"  2. (in a clean clone) python3 data/fetch_benchmark.py  # confirm fetch works")
    print(f"  3. Update SUBMISSION_PROTOCOL.md to point at the new repo id")
    print(f"  4. After upload, retrieve and verify HF's auto-generated Croissant endpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
