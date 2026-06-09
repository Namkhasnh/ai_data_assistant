from __future__ import annotations

from core.rule_generation.base_generator import BaseRuleGenerator
from models.dataset import DatasetMetadata
from models.rule import Rule
from models.semantic_tag import SemanticDetectionReport


class ValidationRuleGenerator(BaseRuleGenerator):
    """Suggest validations for deterministic derived numeric outputs."""

    def __init__(self) -> None:
        super().__init__(name="validation")

    def generate(
        self,
        metadata: DatasetMetadata,
        semantic_report: SemanticDetectionReport,
    ) -> list[Rule]:
        rules: list[Rule] = []
        semantic_by_column = self.semantic_by_column(semantic_report)

        for tag in semantic_by_column.values():
            if tag.semantic_type == "SALARY":
                rules.extend(
                    [
                        self._numeric_validation_rule(f"{tag.column_name}_min", ">", 0),
                        self._numeric_validation_rule(f"{tag.column_name}_max", ">", 0),
                    ]
                )
            elif tag.semantic_type == "EXPERIENCE":
                rules.append(
                    self._numeric_validation_rule(f"{tag.column_name}_years", ">=", 0)
                )

        return rules

    @staticmethod
    def _numeric_validation_rule(
        column: str,
        operator: str,
        value: int,
    ) -> Rule:
        return Rule(
            id=f"rule_validation_{column}_001",
            type="validation",
            column=column,
            parameters={
                "operator": operator,
                "value": value,
                "message": f"{column} must be {operator} {value}",
                "allow_null": True,
            },
            enabled=True,
            priority=40,
            description=f"Validate derived numeric column {column}.",
            created_by="semantic",
        )
