"""Fetch the Survey2Agent reproducibility bundle from Hugging Face.

The extracted bundle is ~400 MB, but the default download path fetches a
single compressed ZIP archive (~36 MB) and expands it locally. This avoids
rate limits that can occur when downloading ~29k small JSON files one by
one from the Hugging Face file tree. The script then fetches the small
top-level metadata files (``README.md``, ``DATA_LICENSE``, ``DATASHEET.md``,
``CITATION.cff``, ``CROISSANT_RAI.json``) separately so reviewers always see
the latest dataset card metadata.

This script downloads the entire bundle into ``$S2A_DATA_ROOT/``
(or ``data/`` if the env var is unset).

Usage::

    pip install huggingface_hub
    python data/fetch_benchmark.py
    # or, with a custom data root:
    S2A_DATA_ROOT=/mnt/fast python data/fetch_benchmark.py
    # or, with a locally downloaded archive:
    python data/fetch_benchmark.py --local-archive bundle.zip

The Hugging Face repo id is read from the ``S2A_HF_REPO`` environment
variable; the default points at the public release.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

# Make the in-tree package importable when this script is run via
# ``python data/fetch_benchmark.py`` from any working directory.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from survey2agent._paths import DATA_ROOT  # noqa: E402

DEFAULT_REPO_ID = "anon-neuripsed26/multisource-memory-benchmark"
DEFAULT_ARCHIVE_PATH = "archives/multisource-memory-benchmark-data-v0.1.0.zip"
DEFAULT_ARCHIVE_SHA256 = (
    "7f1260b1ab6456f46935fa5de582be143a68e19ce6ae5266382b5ee85123a299"
)

REQUIRED_PATHS = (
    "benchmark/seeds",
    "benchmark/results",
    "extracted_atoms/README.md",
    "method_outputs/README.md",
    "README.md",
    "DATA_LICENSE",
    "DATASHEET.md",
    "CITATION.cff",
    "CROISSANT_RAI.json",
)

TOP_LEVEL_METADATA_FILES = (
    "README.md",
    "DATA_LICENSE",
    "DATASHEET.md",
    "CITATION.cff",
    "CROISSANT_RAI.json",
)


def _log(message: str) -> None:
    print(f"[fetch] {message}", flush=True)


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def _heartbeat(label: str, *, interval_s: int = 30):
    """Emit periodic progress while a blocking library call is running."""
    stop = threading.Event()
    started = time.time()

    def run() -> None:
        while not stop.wait(interval_s):
            _log(f"{label} still running after {_fmt_duration(time.time() - started)}")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=0.2)


def _safe_extract_zip(zip_path: Path, target: Path) -> None:
    """Extract a ZIP archive while rejecting paths outside ``target``."""

    target_resolved = target.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise SystemExit(f"Archive integrity check failed at member: {bad}")
        members = zf.infolist()
        _log(f"Extracting {len(members)} archive members into {target} ...")
        started = time.time()
        last = started
        for idx, member in enumerate(members, start=1):
            dest = (target / member.filename).resolve()
            if dest != target_resolved and target_resolved not in dest.parents:
                raise SystemExit(f"Unsafe path in archive: {member.filename}")
            zf.extract(member, path=target)
            now = time.time()
            if idx == 1 or idx == len(members) or now - last >= 30:
                _log(f"  extracted [{idx}/{len(members)}] {member.filename}")
                last = now
        _log(f"Archive extraction completed in {_fmt_duration(time.time() - started)}.")


def validate_bundle_layout(root: Path, *, repo_id: str = DEFAULT_REPO_ID) -> None:
    """Fail loudly if the expected HF snapshot layout is missing.

    ``huggingface_hub.snapshot_download`` may leave an existing local directory
    behind after network or auth failures. A post-download layout check keeps
    ``make fetch`` from reporting success when the reproducibility bundle is
    incomplete.
    """
    _log(f"Validating downloaded layout under {root} ...")
    missing: list[str] = []
    for idx, rel in enumerate(REQUIRED_PATHS, start=1):
        path = root / rel
        if path.exists():
            _log(f"  [{idx}/{len(REQUIRED_PATHS)}] OK      {rel}")
        else:
            _log(f"  [{idx}/{len(REQUIRED_PATHS)}] MISSING {rel}")
            missing.append(rel)
    if missing:
        formatted = "\n".join(f"  - {rel}" for rel in missing)
        raise SystemExit(
            "Download incomplete: the reproducibility bundle is missing:\n"
            f"{formatted}\n\n"
            "Check network access, repository visibility, and S2A_HF_REPO. "
            f"Expected Hugging Face dataset repo: {repo_id}"
        )
    _log("Bundle layout validation passed.")


def fetch_archive_bundle(
    *,
    repo_id: str,
    target: Path,
    archive_path: str = DEFAULT_ARCHIVE_PATH,
    archive_sha256: str = DEFAULT_ARCHIVE_SHA256,
    local_archive: Path | None = None,
) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub is required. Install with:\n"
            "    pip install huggingface_hub"
        ) from exc

    if local_archive is not None:
        zip_path = local_archive
        _log(f"Using local archive: {zip_path}")
    else:
        cache_dir = target / ".cache" / "archives"
        cache_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Downloading compressed archive: {archive_path}")
        _log("This is the recommended path; it avoids downloading ~29k small files.")
        with _heartbeat("archive download"):
            zip_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=archive_path,
                    local_dir=str(cache_dir),
                )
            )

    if not zip_path.exists():
        raise SystemExit(f"Archive not found: {zip_path}")
    actual = _sha256(zip_path)
    _log(f"Archive SHA256: {actual}")
    if actual != archive_sha256:
        raise SystemExit(
            "Archive checksum mismatch.\n"
            f"  expected: {archive_sha256}\n"
            f"  actual  : {actual}\n"
            "Refusing to extract a bundle that does not match the release manifest."
        )

    _safe_extract_zip(zip_path, target)

    _log("Downloading top-level metadata files ...")
    for idx, filename in enumerate(TOP_LEVEL_METADATA_FILES, start=1):
        _log(f"  [{idx}/{len(TOP_LEVEL_METADATA_FILES)}] {filename}")
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(target),
        )


def fetch_snapshot_bundle(*, repo_id: str, target: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "huggingface_hub is required. Install with:\n"
            "    pip install huggingface_hub"
        ) from exc

    _log("Starting Hugging Face snapshot download of the expanded file tree.")
    _log("This fallback can be slower or rate-limited because the release has ~29k files.")
    with _heartbeat("snapshot_download"):
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(target),
        )
    _log(f"Snapshot materialized at {snapshot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("archive", "snapshot"),
        default=os.environ.get("S2A_FETCH_MODE", "archive"),
        help=(
            "Download mode. 'archive' fetches a single release ZIP and is the "
            "default; 'snapshot' downloads the expanded Hugging Face file tree."
        ),
    )
    parser.add_argument(
        "--local-archive",
        type=Path,
        default=Path(os.environ["S2A_LOCAL_ARCHIVE"])
        if os.environ.get("S2A_LOCAL_ARCHIVE")
        else None,
        help="Use a local release ZIP instead of downloading it from Hugging Face.",
    )
    args = parser.parse_args()

    repo_id = os.environ.get("S2A_HF_REPO", DEFAULT_REPO_ID)
    target = DATA_ROOT
    target.mkdir(parents=True, exist_ok=True)

    started = time.time()
    _log(f"Repository : {repo_id}")
    _log(f"Target     : {target}")
    _log(f"Mode       : {args.mode}")
    _log("Expected layout: benchmark/, extracted_atoms/, method_outputs/, "
         "README.md, DATA_LICENSE, DATASHEET.md, CITATION.cff, CROISSANT_RAI.json")
    if args.mode == "archive":
        fetch_archive_bundle(
            repo_id=repo_id,
            target=target,
            local_archive=args.local_archive,
        )
    else:
        if args.local_archive is not None:
            raise SystemExit("--local-archive can only be used with --mode archive")
        fetch_snapshot_bundle(repo_id=repo_id, target=target)
    validate_bundle_layout(target, repo_id=repo_id)
    _log(f"Done in {_fmt_duration(time.time() - started)}.")


if __name__ == "__main__":
    main()
