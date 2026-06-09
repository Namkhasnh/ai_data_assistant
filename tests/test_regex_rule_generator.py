from __future__ import annotations

from core.rule_generation.regex_rule_generator import RegexRuleGenerator
from models.column_profile import ColumnProfile
from models.dataset import DatasetMetadata
from models.semantic_tag import SemanticDetectionReport, SemanticTag


def test_regex_rule_generator_suggests_salary_and_experience_rules() -> None:
    metadata = DatasetMetadata(
        source_file="sample.csv",
        file_format="csv",
        row_count=1,
        column_count=2,
        duplicate_count=0,
        columns=[
            ColumnProfile(
                name="salary",
                data_type="object",
                null_count=0,
                null_percentage=0.0,
                unique_value_count=1,
            ),
            ColumnProfile(
                name="experience",
                data_type="object",
                null_count=0,
                null_percentage=0.0,
                unique_value_count=1,
            ),
        ],
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

    rules = RegexRuleGenerator().generate(metadata, semantic_report)

    assert [rule.id for rule in rules] == [
        "rule_regex_salary_001",
        "rule_regex_experience_001",
    ]
    assert rules[0].parameters["output_columns"] == ["salary_min", "salary_max"]
    assert rules[1].parameters["output_columns"] == ["experience_years"]
    assert all(rule.created_by == "semantic" for rule in rules)
