from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from models.ai_suggestion import AISuggestion


class AIResponseParseError(ValueError):
    """Raised when an LLM response cannot be parsed as suggestion JSON."""


class ResponseParser:
    """Parse strict JSON LLM responses into suggestion objects."""

    def parse(self, raw_response: str) -> list[AISuggestion]:
        """Parse JSON syntax and schema only; business validation happens later."""

        try:
            payload = json.loads(raw_response.strip())
        except json.JSONDecodeError as exc:
            raise AIResponseParseError("AI response must be valid JSON") from exc

        if not isinstance(payload, dict):
            raise AIResponseParseError("AI response must be a JSON object")

        suggestions_payload = payload.get("suggestions")
        if not isinstance(suggestions_payload, list):
            raise AIResponseParseError("AI response must contain suggestions list")

        suggestions: list[AISuggestion] = []
        for item in suggestions_payload:
            suggestions.append(self._parse_suggestion(item))
        return suggestions

    @staticmethod
    def _parse_suggestion(payload: Any) -> AISuggestion:
        try:
            return AISuggestion.model_validate(payload)
        except ValidationError as exc:
            raise AIResponseParseError("AI suggestion has invalid structure") from exc
