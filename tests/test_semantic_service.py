from __future__ import annotations

import json
from pathlib import Path

from core.semantic.base_detector import SemanticRule
from core.semantic.detector_registry import SemanticDetectorRegistry
from core.semantic.keyword_detector import KeywordSemanticDetector
from models.column_profile import ColumnProfile, TopValue
from models.dataset import DatasetMetadata
from services.semantic_service import SemanticService
from storage.artifact_store import ArtifactStore


def test_semantic_service_detects_columns_and_writes_artifact(tmp_path: Path) -> None:
    metadata = DatasetMetadata(
        source_file="jobs.csv",
        file_format="csv",
        row_count=3,
        column_count=4,
        duplicate_count=0,
        columns=[
            ColumnProfile(
                name="title",
                data_type="object",
                null_count=0,
                null_percentage=0.0,
                unique_value_count=2,
                top_values=[TopValue(value="Data Engineer", count=2)],
                sample_values=["AI Engineer"],
            ),
            ColumnProfile(
                name="salary",
                data_type="object",
                null_count=0,
                null_percentage=0.0,
                unique_value_count=1,
                top_values=[TopValue(value="30 - 45 triệu", count=3)],
                sample_values=["Thoả thuận"],
            ),
            ColumnProfile(
                name="location",
                data_type="object",
                null_count=0,
                null_percentage=0.0,
                unique_value_count=2,
                top_values=[TopValue(value="Hồ Chí Minh", count=2)],
                sample_values=["Hà Nội"],
            ),
            ColumnProfile(
                name="unmapped",
                data_type="object",
                null_count=0,
                null_percentage=0.0,
                unique_value_count=1,
            ),
        ],
    )
    artifact_store = ArtifactStore(artifact_dir=tmp_path / "artifacts")

    report = SemanticService(artifact_store=artifact_store).detect_columns(metadata)

    artifact_path = tmp_path / "artifacts" / "semantic_columns.json"
    assert artifact_path.exists()
    assert report.source_file == "jobs.csv"
    assert report.column_count == 4

    semantic_types = {
        column.column_name: column.semantic_type
        for column in report.columns
    }
    assert semantic_types["title"] == "JOB_TITLE"
    assert semantic_types["salary"] == "SALARY"
    assert semantic_types["location"] == "LOCATION"
    assert semantic_types["unmapped"] == "UNKNOWN"

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    first_column = payload["columns"][0]
    assert {
        "semantic_type",
        "confidence",
        "detector_name",
        "evidence",
    }.issubset(first_column)


def test_semantic_service_can_detect_from_metadata_file(tmp_path: Path) -> None:
    metadata = DatasetMetadata(
        source_file="jobs.csv",
        file_format="csv",
        row_count=1,
        column_count=1,
        duplicate_count=0,
        columns=[
            ColumnProfile(
                name="company_name",
                data_type="object",
                null_count=0,
                null_percentage=0.0,
                unique_value_count=1,
                top_values=[TopValue(value="Example Co", count=1)],
            )
        ],
    )
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        metadata.model_dump_json(),
        encoding="utf-8",
    )
    output_path = tmp_path / "semantic_columns.json"

    report = SemanticService().detect_from_metadata_file(
        metadata_path=metadata_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert report.columns[0].semantic_type == "COMPANY"


def test_semantic_service_passes_data_type_to_detectors() -> None:
    registry = SemanticDetectorRegistry(
        detectors=[
            KeywordSemanticDetector(
                rules=[
                    SemanticRule(
                        semantic_type="NUMERIC_IDENTIFIER",
                        keywords=["int64"],
                        confidence=0.77,
                    )
                ]
            )
        ]
    )
    metadata = DatasetMetadata(
        source_file="ids.csv",
        file_format="csv",
        row_count=2,
        column_count=1,
        duplicate_count=0,
        columns=[
            ColumnProfile(
                name="identifier",
                data_type="int64",
                null_count=0,
                null_percentage=0.0,
                unique_value_count=2,
                top_values=[TopValue(value=1, count=1)],
            )
        ],
    )

    report = SemanticService(registry=registry).detect_columns(
        metadata=metadata,
        write_output=False,
    )

    tag = report.columns[0]
    assert tag.semantic_type == "NUMERIC_IDENTIFIER"
    assert tag.detector_name == "keyword_detector"
    assert any("data_type" in evidence for evidence in tag.evidence)
