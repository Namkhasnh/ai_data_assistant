from __future__ import annotations

from core.rule_generation.base_generator import BaseRuleGenerator
from core.rule_generation.generator_registry import GeneratorRegistry
from core.rule_generation.rule_generation_engine import RuleGenerationEngine
from core.rules.rule_registry import RuleRegistry
from models.column_profile import ColumnProfile
from models.dataset import DatasetMetadata
from models.rule import Rule
from models.semantic_tag import SemanticDetectionReport, SemanticTag


class DuplicateGenerator(BaseRuleGenerator):
    def __init__(self) -> None:
        super().__init__(name="duplicate")

    def generate(
        self,
        metadata: DatasetMetadata,
        semantic_report: SemanticDetectionReport,
    ) -> list[Rule]:
        rule = Rule(
            id="rule_duplicate",
            type="transformation",
            column="title",
            parameters={"operation": "trim"},
            priority=20,
            created_by="semantic",
        )
        return [rule, rule]


def test_rule_generation_engine_combines_deduplicates_and_sorts_rules() -> None:
    registry = GeneratorRegistry()
    registry.register("duplicate", DuplicateGenerator)
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
                name="unused",
                data_type="int64",
                null_count=0,
                null_percentage=0.0,
                unique_value_count=1,
            ),
        ],
    )
    semantic_report = SemanticDetectionReport(
        source_file="sample.csv",
        column_count=1,
        columns=[
            SemanticTag(
                column_name="title",
                semantic_type="JOB_TITLE",
                confidence=0.9,
                detector_name="test",
            )
        ],
    )

    rule_set, report = RuleGenerationEngine(
        registry=registry,
        enabled_generators=["duplicate"],
    ).generate(metadata, semantic_report)

    assert len(rule_set.rules) == 1
    assert report.total_generated_rules == 1
    assert report.duplicate_rules_removed == 1
    assert report.generated_by_generator == {"duplicate": 2}
    assert report.skipped_columns == ["unused"]


def test_rule_generation_engine_default_output_is_deterministic() -> None:
    metadata = DatasetMetadata(
        source_file="sample.csv",
        file_format="csv",
        row_count=1,
        column_count=1,
        duplicate_count=0,
        columns=[
            ColumnProfile(
                name="salary",
                data_type="object",
                null_count=0,
                null_percentage=0.0,
                unique_value_count=1,
            )
        ],
    )
    semantic_report = SemanticDetectionReport(
        source_file="sample.csv",
        column_count=1,
        columns=[
            SemanticTag(
                column_name="salary",
                semantic_type="SALARY",
                confidence=0.9,
                detector_name="test",
            )
        ],
    )

    first_rule_set, _ = RuleGenerationEngine().generate(metadata, semantic_report)
    second_rule_set, _ = RuleGenerationEngine().generate(metadata, semantic_report)
    available_rule_types = set(RuleRegistry.default().available_rules())

    assert first_rule_set == second_rule_set
    assert [rule.priority for rule in first_rule_set.rules] == sorted(
        rule.priority for rule in first_rule_set.rules
    )
    assert {
        rule.type
        for rule in first_rule_set.rules
    }.issubset(available_rule_types)
