from __future__ import annotations

from abc import ABC, abstractmethod

from models.dataset import DatasetMetadata
from models.rule import Rule
from models.semantic_tag import SemanticDetectionReport, SemanticTag


class BaseRuleGenerator(ABC):
    """Base interface for deterministic rule suggestion generators."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name
        self.warnings: list[str] = []

    @abstractmethod
    def generate(
        self,
        metadata: DatasetMetadata,
        semantic_report: SemanticDetectionReport,
    ) -> list[Rule]:
        """Generate suggested rules without executing them."""

    @staticmethod
    def semantic_by_column(semantic_report: SemanticDetectionReport) -> dict[str, SemanticTag]:
        return {
            tag.column_name: tag
            for tag in semantic_report.columns
        }
