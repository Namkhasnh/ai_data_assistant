from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from audit.change_log import ArtifactSpec, ChangeLogStore, inspect_artifacts, sha256_file


def test_change_log_is_append_only_and_sorted_by_timestamp(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit" / "change_log.json"
    store = ChangeLogStore(audit_path=audit_path)

    store.record(
        module="standardization",
        artifact="standardization_report.json",
        status="success",
        message="4 columns standardized",
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    store.record(
        module="profiling",
        artifact="metadata.json",
        status="success",
        message="Metadata generated",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    change_log = store.load()

    assert [record.artifact for record in change_log.records] == [
        "metadata.json",
        "standardization_report.json",
    ]
    assert len(change_log.records) == 2

    store.record(
        module="enrichment",
        artifact="enrichment_report.json",
        status="success",
        message="4 columns enriched",
        timestamp=datetime(2026, 1, 3, tzinfo=timezone.utc),
    )

    assert len(store.load().records) == 3


def test_inspect_artifacts_generates_hashes_and_missing_warnings(tmp_path: Path) -> None:
    artifact_path = tmp_path / "metadata.json"
    artifact_path.write_text('{"ok": true}\n', encoding="utf-8")
    original_hash = sha256_file(artifact_path)
    specs = (
        ArtifactSpec("profiling", "metadata.json", artifact_path),
        ArtifactSpec("semantic", "semantic_columns.json", tmp_path / "missing.json"),
    )

    artifacts, warnings = inspect_artifacts(specs)

    assert [artifact.artifact for artifact in artifacts] == [
        "metadata.json",
        "semantic_columns.json",
    ]
    assert artifacts[0].content_hash == original_hash
    assert artifacts[0].size_bytes == artifact_path.stat().st_size
    assert artifacts[1].exists is False
    assert warnings == [f"Missing artifact: {tmp_path / 'missing.json'}"]
    assert sha256_file(artifact_path) == original_hash


def test_change_log_build_from_artifacts_writes_records_without_mutating_sources(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text('{"status": "ok"}\n', encoding="utf-8")
    before_hash = sha256_file(artifact_path)
    store = ChangeLogStore(audit_path=tmp_path / "audit" / "change_log.json")

    change_log, artifacts, warnings = store.build_from_artifacts(
        artifact_specs=(ArtifactSpec("test", "artifact.json", artifact_path),),
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert warnings == []
    assert artifacts[0].content_hash == before_hash
    assert change_log.records[0].status == "success"
    assert sha256_file(artifact_path) == before_hash
