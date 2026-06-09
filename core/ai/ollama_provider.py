from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from core.ai.base_provider import BaseProvider


logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """Ollama text generation provider for local-first LLM suggestions."""

    name = "ollama"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def health_check(self) -> bool:
        """Return False when the local Ollama server is unavailable."""

        request = Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return 200 <= response.status < 300
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return False

    def generate(self, prompt: str) -> str:
        """Generate strict JSON text through Ollama's non-streaming API."""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = self._load_response(response.read())
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Ollama generation failed: {exc}") from exc

        generated_text = response_payload.get("response")
        if not isinstance(generated_text, str):
            raise RuntimeError("Ollama response did not include generated text")
        return generated_text

    @staticmethod
    def _load_response(raw_body: bytes) -> dict[str, Any]:
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Ollama response must be a JSON object")
        return payload
