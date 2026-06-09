from __future__ import annotations

from core.semantic.base_detector import (
    BaseSemanticDetector,
    SemanticDetectionInput,
    SemanticRule,
)
from models.semantic_tag import SemanticTag


class KeywordSemanticDetector(BaseSemanticDetector):
    """Generic detector that matches knowledge-base keywords against column metadata."""

    def __init__(self, rules: list[SemanticRule], name: str = "keyword_detector") -> None:
        super().__init__(rules=rules, name=name)

    def detect(self, detection_input: SemanticDetectionInput) -> SemanticTag | None:
        best_tag: SemanticTag | None = None

        for rule in self.rules:
            if not rule.keywords:
                continue

            evidence = self._collect_evidence(detection_input, rule)
            if not evidence:
                continue

            confidence = min(1.0, rule.confidence + (len(evidence) - 1) * 0.03)
            tag = self.build_tag(
                detection_input=detection_input,
                rule=rule,
                evidence=evidence,
                confidence=confidence,
            )
            if best_tag is None or self.is_better_tag(candidate=tag, current=best_tag):
                best_tag = tag

        return best_tag

    def _collect_evidence(
        self,
        detection_input: SemanticDetectionInput,
        rule: SemanticRule,
    ) -> list[str]:
        evidence: list[str] = []
        normalized_keywords = [
            self.normalize_text(keyword)
            for keyword in rule.keywords
        ]

        for source, value in detection_input.searchable_values():
            normalized_value = self.normalize_text(value)
            for keyword in normalized_keywords:
                if keyword in normalized_value:
                    evidence.append(f"{source} matched keyword '{keyword}'")
                    break

        return evidence
