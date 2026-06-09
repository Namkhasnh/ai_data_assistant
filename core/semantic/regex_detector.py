from __future__ import annotations

import re
from dataclasses import dataclass
from re import Pattern

from core.semantic.base_detector import (
    BaseSemanticDetector,
    SemanticDetectionInput,
    SemanticRule,
)
from models.semantic_tag import SemanticTag


@dataclass(frozen=True)
class CompiledSemanticRule:
    rule: SemanticRule
    patterns: list[Pattern[str]]


class RegexSemanticDetector(BaseSemanticDetector):
    """Generic detector that matches knowledge-base regex patterns."""

    def __init__(self, rules: list[SemanticRule], name: str = "regex_detector") -> None:
        super().__init__(rules=rules, name=name)
        self.compiled_rules = [
            CompiledSemanticRule(
                rule=rule,
                patterns=[re.compile(pattern, flags=re.IGNORECASE) for pattern in rule.patterns],
            )
            for rule in rules
            if rule.patterns
        ]

    def detect(self, detection_input: SemanticDetectionInput) -> SemanticTag | None:
        best_tag: SemanticTag | None = None

        for compiled_rule in self.compiled_rules:
            evidence = self._collect_evidence(detection_input, compiled_rule)
            if not evidence:
                continue

            confidence = min(1.0, compiled_rule.rule.confidence + (len(evidence) - 1) * 0.03)
            tag = self.build_tag(
                detection_input=detection_input,
                rule=compiled_rule.rule,
                evidence=evidence,
                confidence=confidence,
            )
            if best_tag is None or self.is_better_tag(candidate=tag, current=best_tag):
                best_tag = tag

        return best_tag

    def _collect_evidence(
        self,
        detection_input: SemanticDetectionInput,
        compiled_rule: CompiledSemanticRule,
    ) -> list[str]:
        evidence: list[str] = []

        for source, value in detection_input.searchable_values():
            for pattern in compiled_rule.patterns:
                if pattern.search(value):
                    evidence.append(f"{source} matched regex '{pattern.pattern}'")
                    break

        return evidence
