from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.semantic_tag import SemanticTag


UNKNOWN_SEMANTIC_TYPE = "UNKNOWN"


class SemanticDetectionInput(BaseModel):
    """Column metadata used by semantic detectors."""

    model_config = ConfigDict(extra="forbid")

    column_name: str = Field(min_length=1)
    data_type: str | None = None
    sample_values: list[str] = Field(default_factory=list)
    top_values: list[str] = Field(default_factory=list)

    def searchable_values(self) -> Iterable[tuple[str, str]]:
        yield "column_name", self.column_name
        if self.data_type:
            yield "data_type", self.data_type
        for value in self.top_values:
            yield "top_value", value
        for value in self.sample_values:
            yield "sample_value", value


class SemanticRule(BaseModel):
    """Knowledge-base rule consumed by generic semantic detectors."""

    model_config = ConfigDict(extra="forbid")

    semantic_type: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    description: str | None = None

    @field_validator("keywords", "patterns")
    @classmethod
    def _remove_empty_values(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values if value.strip()]


class BaseSemanticDetector(ABC):
    """Base interface for all semantic detectors."""

    name: str

    def __init__(self, rules: list[SemanticRule], name: str) -> None:
        self.rules = rules
        self.name = name

    @abstractmethod
    def detect(self, detection_input: SemanticDetectionInput) -> SemanticTag | None:
        """Return the best semantic tag for this detector, if any."""

    @staticmethod
    def normalize_text(value: str) -> str:
        return " ".join(value.casefold().strip().split())

    def build_tag(
        self,
        detection_input: SemanticDetectionInput,
        rule: SemanticRule,
        evidence: list[str],
        confidence: float | None = None,
    ) -> SemanticTag:
        return SemanticTag(
            column_name=detection_input.column_name,
            semantic_type=rule.semantic_type,
            confidence=confidence if confidence is not None else rule.confidence,
            detector_name=self.name,
            evidence=evidence,
        )

    @classmethod
    def is_better_tag(cls, candidate: SemanticTag, current: SemanticTag) -> bool:
        candidate_rank = cls.evidence_rank(candidate)
        current_rank = cls.evidence_rank(current)
        if candidate_rank != current_rank:
            return candidate_rank > current_rank
        return candidate.confidence > current.confidence

    @staticmethod
    def evidence_rank(tag: SemanticTag) -> int:
        evidence_text = " ".join(tag.evidence)
        if "column_name" in evidence_text:
            return 4
        if "data_type" in evidence_text:
            return 3
        if "top_value" in evidence_text:
            return 2
        if "sample_value" in evidence_text:
            return 1
        return 0
