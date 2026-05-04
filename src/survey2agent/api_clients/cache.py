"""SHA256-keyed disk cache for LLM completions.

Layout:
    {root}/{provider}/{api_model_id}/{sha256}.json

The hash is computed from a canonical JSON serialization of the request
payload (see `CompletionRequest.to_cache_payload()`) plus the bound
`model_id` plus `CACHE_KEY_VERSION`. See `CACHE_POLICY.md` at the repo root
for the user-facing description.

The on-disk cache is intentionally slim: each file stores only the response
text and the model identifier. No tokens, no costs, no provider payloads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .types import CompletionRequest, CompletionResult


# Bumping this constant invalidates every existing cache entry. Bump whenever:
#   - a new field is added to CompletionRequest's cache payload
#   - the canonical serialization rule in `compute_cache_key` changes
CACHE_KEY_VERSION = "v2"

# Cache file schema marker. Independent of the key version: bumping this only
# changes file parsing, not the keys.
CACHE_FILE_VERSION = "v2"


class CacheMissError(RuntimeError):
    """Raised by `SyncLLMClient.complete()` when no cache entry exists and
    `allow_api_call=False`."""

    def __init__(self, key: str, provider: str, model_id: str, hint: str = "") -> None:
        msg = (
            f"cache miss for key {key} (provider={provider}, model_id={model_id}). "
            "Pass allow_api_call=True to fetch live, or verify the cache "
            "directory contains the expected file."
        )
        if hint:
            msg += f"\nhint: {hint}"
        super().__init__(msg)
        self.key = key
        self.provider = provider
        self.model_id = model_id


def compute_cache_key(request: CompletionRequest, model_id: str) -> str:
    """Deterministic SHA256 of (request payload, model_id, version).

    The output is hex-encoded (64 chars). Independent of dict iteration order
    and Python version (we use `sort_keys=True` and explicit field ordering
    in `CompletionRequest.to_cache_payload`).
    """
    payload = {
        "request": request.to_cache_payload(),
        "model_id": model_id,
        "version": CACHE_KEY_VERSION,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class CacheStore:
    """File-system cache for `CompletionResult` objects.

    Single instance per `cache_dir`; thread-safe writes are NOT guaranteed
    (writes are atomic per-file via os.replace, but concurrent writes to the
    SAME key could race). For the paper pipeline we never write the same key
    concurrently, so this is acceptable.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------
    # Path
    # ------------------------------------------------------------------

    def _path(self, key: str, provider: str, api_model_id: str) -> Path:
        # api_model_id may contain '/' on some providers (OpenRouter) — sanitize
        safe_model = api_model_id.replace("/", "__")
        return self.root / provider / safe_model / f"{key}.json"

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def get(self, key: str, provider: str, api_model_id: str) -> CompletionResult | None:
        path = self._path(key, provider, api_model_id)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        return _from_cache_record(record, provider=provider, model_id=api_model_id)

    def put(self, key: str, result: CompletionResult) -> None:
        path = self._path(key, result.provider, result.model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = _to_cache_record(result)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
        tmp.replace(path)


# ---------------------------------------------------------------------------
# Cache file (de)serialization
# ---------------------------------------------------------------------------

def _to_cache_record(result: CompletionResult) -> dict[str, Any]:
    return {
        "cache_file_version": CACHE_FILE_VERSION,
        "model": result.model_id,
        "response": result.text,
    }


def _from_cache_record(
    record: dict[str, Any],
    *,
    provider: str,
    model_id: str,
) -> CompletionResult:
    """Parse an on-disk record back into a `CompletionResult` (cache_hit=True).

    Required schema:
        {"cache_file_version": "v2", "model": str, "response": str}
    """
    version = record.get("cache_file_version")
    if version != CACHE_FILE_VERSION:
        raise ValueError(
            f"unsupported cache file schema: cache_file_version={version!r}, "
            f"expected {CACHE_FILE_VERSION!r}"
        )
    cached_model = str(record.get("model", model_id))
    text = str(record["response"])
    return CompletionResult(
        text=text,
        finish_reason=None,
        model_id=cached_model,
        provider=provider,
        cache_hit=True,
    )
