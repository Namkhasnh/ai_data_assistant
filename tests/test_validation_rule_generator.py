from __future__ import annotations

from core.rule_generation.validation_rule_generator import ValidationRuleGenerator
from models.dataset import DatasetMetadata
from models.semantic_tag import SemanticDetectionReport, SemanticTag


def test_validation_rule_generator_suggests_rules_for_derived_numeric_outputs() -> None:
    metadata = DatasetMetadata(
        source_file="sample.csv",
        file_format="csv",
        row_count=1,
        column_count=0,
        duplicate_count=0,
    )
    semantic_report = SemanticDetectionReport(
        source_file="sample.csv",
        column_count=2,
        columns=[
            SemanticTag(
                column_name="salary",
                semantic_type="SALARY",
                confidence=0.9,
                detector_name="test",
            ),
            SemanticTag(
                column_name="experience",
                semantic_type="EXPERIENCE",
                confidence=0.9,
                detector_name="test",
            ),
        ],
    )

    rules = ValidationRuleGenerator().generate(metadata, semantic_report)

    assert [rule.column for rule in rules] == [
        "salary_min",
        "salary_max",
        "experience_years",
    ]
    assert rules[0].parameters["operator"] == ">"
    assert rules[2].parameters["operator"] == ">="
    assert all(rule.parameters["allow_null"] is True for rule in rules)
    assert all(rule.type == "validation" for rule in rules)
