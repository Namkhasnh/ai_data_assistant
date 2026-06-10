from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from services.report_service import ReportService


def test_report_service_generates_report_assets_without_mutating_inputs(tmp_path):
    artifact_dir = tmp_path / "storage" / "artifacts"
    audit_dir = tmp_path / "storage" / "audit"
    rules_dir = tmp_path / "data" / "rules"
    report_dir = tmp_path / "storage" / "reports"
    artifact_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)
    rules_dir.mkdir(parents=True)

    _write_json(
        artifact_dir / "metadata.json",
        {
            "source_file": "sample.csv",
            "file_format": "csv",
            "row_count": 3,
            "column_count": 3,
            "duplicate_count": 0,
            "columns": [
                _column("row_id", "int64", 0, 3),
                _column("title", "object", 0, 2),
                _column("location", "object", 0, 2),
            ],
        },
    )
    _write_json(
        artifact_dir / "semantic_columns.json",
        {
            "source_file": "sample.csv",
            "column_count": 3,
            "columns": [
                {
                    "column_name": "title",
                    "semantic_type": "JOB_TITLE",
                    "confidence": 0.9,
                    "detector_name": "keyword_detector",
                    "evidence": [],
                },
                {
                    "column_name": "location",
                    "semantic_type": "LOCATION",
                    "confidence": 0.8,
                    "detector_name": "keyword_detector",
                    "evidence": [],
                },
            ],
        },
    )
    _write_json(
        rules_dir / "rules.json",
        {
            "rules": [
                {
                    "id": "rule_001",
                    "type": "mapping",
                    "column": "title",
                    "parameters": {},
                    "enabled": True,
                    "priority": 10,
                    "created_by": "semantic",
                }
            ]
        },
    )
    _write_json(
        artifact_dir / "rule_execution_report.json",
        {
            "results": [
                {
                    "rule_id": "rule_001",
                    "rule_type": "mapping",
                    "status": "applied",
                    "affected_rows": 2,
                    "execution_time_ms": 1.0,
                    "message": "Rule applied",
                }
            ]
        },
    )
    _write_json(
        artifact_dir / "standardization_report.json",
        {
            "total_standardized_columns": 2,
            "standardized_by_standardizer": {"mapping": 2},
            "warnings": [],
            "skipped_columns": [],
            "standardizer_warnings": {},
        },
    )
    _write_json(
        artifact_dir / "enrichment_report.json",
        {
            "total_enriched_columns": 2,
            "enriched_by_enricher": {"knowledge": 2},
            "warnings": [],
            "skipped_columns": [],
            "enricher_warnings": {},
        },
    )
    _write_json(
        artifact_dir / "ai_suggestions.json",
        {
            "suggestions": [],
            "warnings": ["provider unavailable"],
            "provider": "ollama",
            "model": "qwen3:8b",
            "minimum_confidence": 0.0,
            "duplicate_suggestions_removed": 0,
            "rejected_suggestions_count": 0,
        },
    )
    _write_json(
        audit_dir / "audit_report.json",
        {
            "generated_at": "2026-06-10T00:00:00Z",
            "warnings": [],
            "artifacts": [
                {
                    "artifact": "metadata.json",
                    "path": "storage/artifacts/metadata.json",
                    "module": "profiling",
                    "exists": True,
                    "content_hash": "abc",
                    "size_bytes": 100,
                    "modified_at": "2026-06-10T00:00:00Z",
                }
            ],
            "change_log": {"records": []},
            "rule_history": {"records": []},
        },
    )
    pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "title": ["Business Data Analyst", "Unknown Title", "Data Engineer"],
            "location": ["Ha Noi", "Da Nang", "Ho Chi Minh"],
            "standardized_title": ["Data Analyst", None, "Data Engineer"],
            "standardized_location": ["Ha Noi", "Da Nang", "Ho Chi Minh"],
            "job_family": ["Analytics", None, "Engineering"],
            "region": ["North", None, "South"],
        }
    ).to_csv(artifact_dir / "enriched_dataset.csv", index=False)

    input_files = [
        artifact_dir / "metadata.json",
        artifact_dir / "semantic_columns.json",
        rules_dir / "rules.json",
        artifact_dir / "rule_execution_report.json",
        artifact_dir / "standardization_report.json",
        artifact_dir / "enrichment_report.json",
        artifact_dir / "ai_suggestions.json",
        audit_dir / "audit_report.json",
        artifact_dir / "enriched_dataset.csv",
    ]
    before_hashes = {path: _sha256(path) for path in input_files}

    report, output_path = ReportService(
        artifact_dir=artifact_dir,
        audit_dir=audit_dir,
        rules_path=rules_dir / "rules.json",
        report_dir=report_dir,
    ).generate_report()

    assert output_path.exists()
    assert (report_dir / "assets" / "semantic_distribution.html").exists()
    assert (report_dir / "assets" / "rule_distribution.html").exists()
    assert (report_dir / "assets" / "artifact_sizes.html").exists()
    assert "numeric-row-id.html" in {path.name for path in (report_dir / "assets").glob("*.html")}
    assert before_hashes == {path: _sha256(path) for path in input_files}
    assert report.warnings == []


def test_report_service_warns_and_continues_when_artifacts_are_missing(tmp_path):
    report, output_path = ReportService(
        artifact_dir=tmp_path / "missing_artifacts",
        audit_dir=tmp_path / "missing_audit",
        rules_path=tmp_path / "missing_rules" / "rules.json",
        report_dir=tmp_path / "reports",
    ).generate_report()

    assert output_path.exists()
    assert report.warnings
    assert any("Missing artifact" in warning for warning in report.warnings)
    assert "Dataset metadata is unavailable" in output_path.read_text(encoding="utf-8")


def _column(name: str, data_type: str, null_count: int, unique_count: int) -> dict[str, Any]:
    return {
        "name": name,
        "data_type": data_type,
        "null_count": null_count,
        "null_percentage": 0.0,
        "unique_value_count": unique_count,
        "top_values": [],
        "sample_values": [],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
