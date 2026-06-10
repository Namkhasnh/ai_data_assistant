from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from audit.audit_report import AuditReportStore
from audit.change_log import ArtifactSpec, sha256_file


def test_audit_report_persists_without_mutating_existing_audit_files(tmp_path: Path) -> None:
    change_log_path = tmp_path / "change_log.json"
    rule_history_path = tmp_path / "rule_history.json"
    audit_report_path = tmp_path / "audit_report.json"
    artifact_a = tmp_path / "b_artifact.json"
    artifact_b = tmp_path / "a_artifact.json"
    artifact_a.write_text('{"artifact": "b"}\n', encoding="utf-8")
    artifact_b.write_text('{"artifact": "a"}\n', encoding="utf-8")
    change_log_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "timestamp": "2026-01-02T00:00:00Z",
                        "module": "module_b",
                        "artifact": "b_artifact.json",
                        "status": "success",
                        "message": "second",
                    },
                    {
                        "timestamp": "2026-01-01T00:00:00Z",
                        "module": "module_a",
                        "artifact": "a_artifact.json",
                        "status": "success",
                        "message": "first",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    rule_history_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "rule_id": "rule_b",
                        "rule_type": "regex",
                        "created_by": "semantic",
                        "enabled": True,
                        "execution_count": 1,
                        "first_seen": "2026-01-01T00:00:00Z",
                        "last_seen": "2026-01-02T00:00:00Z",
                    },
                    {
                        "rule_id": "rule_a",
                        "rule_type": "mapping",
                        "created_by": "manual",
                        "enabled": False,
                        "execution_count": 0,
                        "first_seen": "2026-01-01T00:00:00Z",
                        "last_seen": "2026-01-01T00:00:00Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    change_hash = sha256_file(change_log_path)
    history_hash = sha256_file(rule_history_path)

    report, output_path = AuditReportStore(
        audit_path=audit_report_path,
        change_log_path=change_log_path,
        rule_history_path=rule_history_path,
    ).persist(
        artifact_specs=(
            ArtifactSpec("module_b", "b_artifact.json", artifact_a),
            ArtifactSpec("module_a", "a_artifact.json", artifact_b),
        )
    )

    assert output_path == audit_report_path
    assert audit_report_path.exists()
    assert [artifact.artifact for artifact in report.artifacts] == [
        "a_artifact.json",
        "b_artifact.json",
    ]
    assert [record.artifact for record in report.change_log.records] == [
        "a_artifact.json",
        "b_artifact.json",
    ]
    assert [record.rule_id for record in report.rule_history.records] == [
        "rule_a",
        "rule_b",
    ]
    assert sha256_file(change_log_path) == change_hash
    assert sha256_file(rule_history_path) == history_hash


def test_audit_report_records_missing_file_warnings(tmp_path: Path) -> None:
    report = AuditReportStore(
        audit_path=tmp_path / "audit_report.json",
        change_log_path=tmp_path / "missing_change_log.json",
        rule_history_path=tmp_path / "missing_rule_history.json",
    ).build(
        artifact_specs=(
            ArtifactSpec("missing", "missing.json", tmp_path / "missing.json"),
        )
    )

    assert report.artifacts[0].exists is False
    assert any("Missing artifact" in warning for warning in report.warnings)
    assert any("Missing change log" in warning for warning in report.warnings)
    assert any("Missing rule history" in warning for warning in report.warnings)


def test_audit_report_serializes_generated_at_as_utc(tmp_path: Path) -> None:
    report, _ = AuditReportStore(
        audit_path=tmp_path / "audit_report.json",
        change_log_path=tmp_path / "missing_change_log.json",
        rule_history_path=tmp_path / "missing_rule_history.json",
    ).persist(artifact_specs=())
    payload = report.model_dump(mode="json")

    assert payload["generated_at"].endswith("Z")
    datetime.fromisoformat(payload["generated_at"].replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
