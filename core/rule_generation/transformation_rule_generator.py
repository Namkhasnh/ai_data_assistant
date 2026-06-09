from __future__ import annotations

from core.rule_generation.base_generator import BaseRuleGenerator
from models.dataset import DatasetMetadata
from models.rule import Rule
from models.semantic_tag import SemanticDetectionReport


class TransformationRuleGenerator(BaseRuleGenerator):
    """Suggest safe, non-destructive text-cleaning transformations."""

    TEXT_DTYPES = {"object", "string", "str", "utf8"}

    def __init__(self) -> None:
        super().__init__(name="transformation")

    def generate(
        self,
        metadata: DatasetMetadata,
        semantic_report: SemanticDetectionReport,
    ) -> list[Rule]:
        rules: list[Rule] = []
        for column in metadata.columns:
            if column.data_type.casefold() not in self.TEXT_DTYPES:
                continue

            rules.extend(
                [
                    Rule(
                        id=f"rule_transform_{column.name}_trim_001",
                        type="transformation",
                        column=column.name,
                        parameters={
                            "operation": "trim",
                        },
                        enabled=True,
                        priority=10,
                        description=f"Trim leading and trailing whitespace in {column.name}.",
                        created_by="semantic",
                    ),
                    Rule(
                        id=f"rule_transform_{column.name}_remove_extra_spaces_001",
                        type="transformation",
                        column=column.name,
                        parameters={
                            "operation": "remove_extra_spaces",
                        },
                        enabled=True,
                        priority=11,
                        description=f"Collapse repeated spaces in {column.name}.",
                        created_by="semantic",
                    ),
                ]
            )
        return rules
