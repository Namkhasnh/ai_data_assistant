from __future__ import annotations

import json
from pathlib import Path

from core.rule_generation.mapping_rule_generator import MappingRuleGenerator
from models.column_profile import ColumnProfile, TopValue
from models.dataset import DatasetMetadata
from models.semantic_tag import SemanticDetectionReport, SemanticTag


def test_mapping_rule_generator_uses_deterministic_knowledge(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "cities.json").write_text(
        json.dumps(
            {
                "Hanoi": {
                    "aliases": ["HN", "Ha Noi", "Hà Nội"],
                }
            }
        ),
        encoding="utf-8",
    )
    metadata = DatasetMetadata(
        source_file="sample.csv",
        file_format="csv",
        row_count=3,
        column_count=1,
        duplicate_count=0,
        columns=[
            ColumnProfile(
                name="location",
                data_type="object",
                null_count=0,
                null_percentage=0.0,
                unique_value_count=3,
                top_values=[TopValue(value="HN", count=2)],
                sample_values=["Ha Noi"],
            )
        ],
    )
    semantic_report = SemanticDetectionReport(
        source_file="sample.csv",
        column_count=1,
        columns=[
            SemanticTag(
                column_name="location",
                semantic_type="LOCATION",
                confidence=0.9,
                detector_name="test",
            )
        ],
    )

    rules = MappingRuleGenerator(knowledge_dir=knowledge_dir).generate(metadata, semantic_report)

    assert len(rules) == 1
    assert rules[0].id == "rule_mapping_location_001"
    assert rules[0].type == "mapping"
    assert rules[0].parameters["mapping"] == {
        "HN": "Hanoi",
        "Ha Noi": "Hanoi",
    }
    assert rules[0].created_by == "semantic"


def test_mapping_rule_generator_does_not_invent_mappings_without_knowledge(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "cities.json").write_text("", encoding="utf-8")
    metadata = DatasetMetadata(
        source_file="sample.csv",
        file_format="csv",
        row_count=1,
        column_count=1,
        duplicate_count=0,
        columns=[
            ColumnProfile(
                name="location",
                data_type="object",
                null_count=0,
                null_percentage=0.0,
                unique_value_count=1,
                sample_values=["HN"],
            )
        ],
    )
    semantic_report = SemanticDetectionReport(
        source_file="sample.csv",
        column_count=1,
        columns=[
            SemanticTag(
                column_name="location",
                semantic_type="LOCATION",
                confidence=0.9,
                detector_name="test",
            )
        ],
    )
    generator = MappingRuleGenerator(knowledge_dir=knowledge_dir)

    rules = generator.generate(metadata, semantic_report)

    assert rules == []
    assert generator.warnings
