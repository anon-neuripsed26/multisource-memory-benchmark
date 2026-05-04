"""Custom pytest markers and data-availability skip hook."""

from __future__ import annotations

import pytest

from survey2agent._paths import DATA_ROOT

_NEEDS_DATA_REASON = (
    "requires data — run 'make fetch' or set S2A_DATA_ROOT to a populated bundle"
)


def _data_available() -> bool:
    """Return True iff the real reproducibility bundle is present.

    We probe the three top-level directories that ship together in the
    Hugging Face bundle.  When the repo is freshly cloned from the 4open /
    GitHub mirror, these are absent and every ``needs_data`` test is
    auto-skipped.
    """
    return all(
        (DATA_ROOT / name).is_dir()
        for name in ("benchmark", "extracted_atoms", "method_outputs")
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_api: test that requires LLM API access (skipped by default in CI)",
    )
    config.addinivalue_line(
        "markers",
        "needs_data: test that requires the full reproducibility data bundle "
        "(auto-skipped when data/ is empty; run 'make fetch' to enable)",
    )


def pytest_collection_modifyitems(config, items):
    if _data_available():
        return
    skip_marker = pytest.mark.skip(reason=_NEEDS_DATA_REASON)
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip_marker)
