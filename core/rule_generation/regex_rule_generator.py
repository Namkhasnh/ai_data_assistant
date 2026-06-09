from __future__ import annotations

from core.rule_generation.base_generator import BaseRuleGenerator
from models.dataset import DatasetMetadata
from models.rule import Rule
from models.semantic_tag import SemanticDetectionReport


class RegexRuleGenerator(BaseRuleGenerator):
    """Suggest deterministic extraction rules from semantic types."""

    def __init__(self) -> None:
        super().__init__(name="regex")

    def generate(
        self,
        metadata: DatasetMetadata,
        semantic_report: SemanticDetectionReport,
    ) -> list[Rule]:
        semantic_by_column = self.semantic_by_column(semantic_report)
        rules: list[Rule] = []

        for column in metadata.columns:
            semantic_tag = semantic_by_column.get(column.name)
            if semantic_tag is None:
                continue

            if semantic_tag.semantic_type == "SALARY":
                rules.append(
                    Rule(
                        id=f"rule_regex_{column.name}_001",
                        type="regex",
                        column=column.name,
                        parameters={
                            "pattern": r"(\d+)\s*-\s*(\d+)",
                            "output_columns": [
                                f"{column.name}_min",
                                f"{column.name}_max",
                            ],
                            "output_types": {
                                f"{column.name}_min": "int",
                                f"{column.name}_max": "int",
                            },
                        },
                        enabled=True,
                        priority=30,
                        description=f"Extract numeric range from {column.name}.",
                        created_by="semantic",
                    )
                )
            elif semantic_tag.semantic_type == "EXPERIENCE":
                rules.append(
                    Rule(
                        id=f"rule_regex_{column.name}_001",
                        type="regex",
                        column=column.name,
                        parameters={
                            "pattern": r"(\d+)",
                            "output_columns": [
                                f"{column.name}_years",
                            ],
                            "output_types": {
                                f"{column.name}_years": "int",
                            },
                        },
                        enabled=True,
                        priority=30,
                        description=f"Extract numeric years from {column.name}.",
                        created_by="semantic",
                    )
                )

        return rules
