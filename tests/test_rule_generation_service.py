from __future__ import annotations

import json
from pathlib import Path

from models.column_profile import ColumnProfile
from models.dataset import DatasetMetadata
from models.semantic_tag import SemanticDetectionReport, SemanticTag
from services.rule_generation_service import RuleGenerationService
from storage.artifact_store import ArtifactStore


def test_rule_generation_service_loads_artifacts_and_writes_outputs(tmp_path: Path) -> None:
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
    metadata_path = tmp_path / "metadata.json"
    semantic_path = tmp_path / "semantic_columns.json"
    rules_output_path = tmp_path / "rules" / "rules.json"
    metadata_path.write_text(metadata.model_dump_json(), encoding="utf-8")
    semantic_path.write_text(semantic_report.model_dump_json(), encoding="utf-8")

    rule_set, report = RuleGenerationService(
        artifact_store=ArtifactStore(artifact_dir=tmp_path / "artifacts")
    ).generate_from_artifacts(
        metadata_path=metadata_path,
        semantic_report_path=semantic_path,
        rules_output_path=rules_output_path,
    )

    assert rules_output_path.exists()
    assert (tmp_path / "artifacts" / "rule_generation_report.json").exists()
    assert not (tmp_path / "artifacts" / "cleaned_dataset.csv").exists()
    assert not (tmp_path / "artifacts" / "rule_execution_report.json").exists()
    assert rule_set.rules
    assert report.total_generated_rules == len(rule_set.rules)

    payload = json.loads(rules_output_path.read_text(encoding="utf-8"))
    assert payload["rules"][0]["created_by"] == "semantic"
