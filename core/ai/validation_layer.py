from __future__ import annotations

import hashlib
import json
from typing import Any

from models.ai_suggestion import AISuggestion, AISuggestionReport


class ValidationLayer:
    """Validate LLM suggestions before persistence or human review."""

    def __init__(
        self,
        allowed_types: set[str] | None = None,
        minimum_confidence: float = 0.0,
    ) -> None:
        self.allowed_types = allowed_types or {
            "mapping",
            "regex",
            "transformation",
            "validation",
        }
        self.minimum_confidence = minimum_confidence

    def validate(
        self,
        suggestions: list[AISuggestion],
        provider: str | None = None,
        model: str | None = None,
        warnings: list[str] | None = None,
    ) -> AISuggestionReport:
        """Return a validated report with duplicates and invalid suggestions removed."""

        report_warnings = list(warnings or [])
        valid_suggestions: list[AISuggestion] = []
        seen_keys: set[str] = set()
        duplicate_count = 0
        rejected_count = 0

        for suggestion in suggestions:
            normalized = self._normalize_suggestion(suggestion)
            rejection_reason = self._rejection_reason(normalized)
            if rejection_reason is not None:
                rejected_count += 1
                report_warnings.append(rejection_reason)
                continue

            duplicate_key = self._duplicate_key(normalized)
            if duplicate_key in seen_keys:
                duplicate_count += 1
                continue

            seen_keys.add(duplicate_key)
            valid_suggestions.append(
                normalized.model_copy(
                    update={"suggestion_id": normalized.suggestion_id or self._suggestion_id(normalized)}
                )
            )

        if duplicate_count:
            report_warnings.append(f"Removed {duplicate_count} duplicate AI suggestion(s)")

        return AISuggestionReport(
            suggestions=valid_suggestions,
            warnings=report_warnings,
            provider=provider,
            model=model,
            minimum_confidence=self.minimum_confidence,
            duplicate_suggestions_removed=duplicate_count,
            rejected_suggestions_count=rejected_count,
        )

    def empty_report(
        self,
        provider: str | None = None,
        model: str | None = None,
        warnings: list[str] | None = None,
    ) -> AISuggestionReport:
        """Return a valid empty report for non-fatal provider or parser failures."""

        return AISuggestionReport(
            suggestions=[],
            warnings=list(warnings or []),
            provider=provider,
            model=model,
            minimum_confidence=self.minimum_confidence,
        )

    def _normalize_suggestion(self, suggestion: AISuggestion) -> AISuggestion:
        return suggestion.model_copy(
            update={
                "type": suggestion.type.strip().lower(),
                "column": self._normalize_optional_text(suggestion.column),
                "semantic_type": self._normalize_optional_text(suggestion.semantic_type),
                "source_value": self._normalize_optional_text(suggestion.source_value),
                "target_value": self._normalize_optional_text(suggestion.target_value),
                "created_by": suggestion.created_by.strip().lower(),
                "reason": suggestion.reason.strip(),
            }
        )

    def _rejection_reason(self, suggestion: AISuggestion) -> str | None:
        if suggestion.type not in self.allowed_types:
            return f"Rejected unsupported AI suggestion type: {suggestion.type}"
        if suggestion.confidence < 0.0 or suggestion.confidence > 1.0:
            return f"Rejected AI suggestion with invalid confidence: {suggestion.confidence}"
        if suggestion.confidence < self.minimum_confidence:
            return (
                "Rejected AI suggestion below minimum confidence "
                f"{self.minimum_confidence}: {suggestion.confidence}"
            )
        if suggestion.created_by != "llm":
            return f"Rejected AI suggestion with unsupported creator: {suggestion.created_by}"
        if suggestion.type == "mapping" and (
            not suggestion.source_value or not suggestion.target_value
        ):
            return "Rejected mapping suggestion without source_value and target_value"
        return None

    @staticmethod
    def _normalize_optional_text(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _duplicate_key(suggestion: AISuggestion) -> str:
        return json.dumps(
            {
                "type": suggestion.type,
                "column": suggestion.column,
                "semantic_type": suggestion.semantic_type,
                "source_value": suggestion.source_value,
                "target_value": suggestion.target_value,
                "parameters": suggestion.parameters,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    @classmethod
    def _suggestion_id(cls, suggestion: AISuggestion) -> str:
        digest = hashlib.sha256(cls._duplicate_key(suggestion).encode("utf-8")).hexdigest()
        return f"ai_suggestion_{digest[:12]}"
