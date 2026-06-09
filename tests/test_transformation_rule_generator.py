from __future__ import annotations

from core.rule_generation.transformation_rule_generator import TransformationRuleGenerator
from models.column_profile import ColumnProfile
from models.dataset import DatasetMetadata
from models.semantic_tag import SemanticDetectionReport


def test_transformation_rule_generator_suggests_safe_text_cleaning_only() -> None:
    metadata = DatasetMetadata(
        source_file="sample.csv",
        file_format="csv",
        row_count=1,
        column_count=2,
        duplicate_count=0,
        columns=[
            ColumnProfile(
                name="title",
                data_type="object",
                null_count=0,
                null_percentage=0.0,
                unique_value_count=1,
            ),
            ColumnProfile(
                name="job_id",
                data_type="int64",
                null_count=0,
                null_percentage=0.0,
                unique_value_count=1,
            ),
        ],
    )
    semantic_report = SemanticDetectionReport(source_file="sample.csv", column_count=0)

    rules = TransformationRuleGenerator().generate(metadata, semantic_report)

    assert [rule.parameters["operation"] for rule in rules] == [
        "trim",
        "remove_extra_spaces",
    ]
    assert all(rule.column == "title" for rule in rules)
    assert all(rule.type == "transformation" for rule in rules)
    assert all(rule.enabled for rule in rules)
