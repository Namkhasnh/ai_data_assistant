from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.semantic.base_detector import (
    BaseSemanticDetector,
    SemanticDetectionInput,
    SemanticRule,
    UNKNOWN_SEMANTIC_TYPE,
)
from core.semantic.keyword_detector import KeywordSemanticDetector
from core.semantic.regex_detector import RegexSemanticDetector
from models.semantic_tag import SemanticTag


class SemanticKnowledgeBase(BaseModel):
    """Knowledge file schema for semantic detection rules."""

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=1, ge=1)
    rules: list[SemanticRule] = Field(default_factory=list)


class SemanticDetectorRegistry:
    """Registry and orchestrator for semantic detectors."""

    def __init__(self, detectors: list[BaseSemanticDetector] | None = None) -> None:
        self.detectors = detectors or []

    def register(self, detector: BaseSemanticDetector) -> None:
        self.detectors.append(detector)

    def detect_column(self, detection_input: SemanticDetectionInput) -> SemanticTag:
        best_tag: SemanticTag | None = None

        for detector in self.detectors:
            tag = detector.detect(detection_input)
            if tag is None:
                continue
            if best_tag is None or self._is_better_tag(candidate=tag, current=best_tag):
                best_tag = tag

        if best_tag is None:
            return SemanticTag(
                column_name=detection_input.column_name,
                semantic_type=UNKNOWN_SEMANTIC_TYPE,
                confidence=0.0,
                detector_name="detector_registry",
                evidence=[],
            )

        return best_tag

    @classmethod
    def _is_better_tag(cls, candidate: SemanticTag, current: SemanticTag) -> bool:
        candidate_rank = BaseSemanticDetector.evidence_rank(candidate)
        current_rank = BaseSemanticDetector.evidence_rank(current)
        if candidate_rank != current_rank:
            return candidate_rank > current_rank
        return candidate.confidence > current.confidence

    def detect_many(
        self,
        detection_inputs: list[SemanticDetectionInput],
    ) -> list[SemanticTag]:
        return [
            self.detect_column(detection_input)
            for detection_input in detection_inputs
        ]

    @classmethod
    def from_knowledge_file(cls, knowledge_path: str | Path) -> SemanticDetectorRegistry:
        knowledge_base = cls._load_knowledge_base(Path(knowledge_path))
        registry = cls()
        registry.register(KeywordSemanticDetector(rules=knowledge_base.rules))
        registry.register(RegexSemanticDetector(rules=knowledge_base.rules))
        return registry

    @classmethod
    def from_knowledge_dir(cls, knowledge_dir: str | Path = "knowledge") -> SemanticDetectorRegistry:
        return cls.from_knowledge_file(Path(knowledge_dir) / "semantic_rules.json")

    @staticmethod
    def _load_knowledge_base(knowledge_path: Path) -> SemanticKnowledgeBase:
        try:
            payload: dict[str, Any] = json.loads(knowledge_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Semantic knowledge file not found: {knowledge_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Semantic knowledge file is invalid JSON: {knowledge_path}") from exc

        return SemanticKnowledgeBase.model_validate(payload)
