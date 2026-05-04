"""OpenRouter sync client (DeepSeek V3.2, Qwen3 235B).

OpenRouter exposes an OpenAI-compatible chat completions endpoint. Each call
is a single HTTP request; OpenRouter does not expose a batch API.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

import httpx

from .base import SyncLLMClient
from .types import CompletionRequest


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterSyncClient(SyncLLMClient):
    provider: ClassVar[str] = "openrouter"
    allowed_request_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "user_prompt",
            "system_prompt",
            "temperature",
            "top_p",
            "seed",
        }
    )

    def _api_key(self) -> str:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Either set it or run with allow_api_call=False to use cached responses only."
            )
        return key

    def _raw_complete(self, request: CompletionRequest) -> str:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.user_prompt})

        body: dict[str, Any] = {
            "model": self.model_config.api_model_id,
            "messages": messages,
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.seed is not None:
            body["seed"] = request.seed

        url = self.model_config.api_endpoint or f"{OPENROUTER_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
            response = client.post(url, json=body, headers=headers)
            response.raise_for_status()
            payload = response.json()

        choices = payload.get("choices") or []
        if not choices:
            raise ValueError("OpenRouter response has no choices")
        return (choices[0].get("message") or {}).get("content") or ""
