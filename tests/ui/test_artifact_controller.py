from __future__ import annotations

import json

from app.controllers.artifact_controller import ArtifactController


def test_artifact_controller_lists_exports_reports_and_audit_files(tmp_path) -> None:
    export_dir = tmp_path / "exports"
    report_dir = tmp_path / "reports"
    audit_dir = tmp_path / "audit"
    (export_dir / "csv").mkdir(parents=True)
    report_dir.mkdir()
    audit_dir.mkdir()
    (export_dir / "csv" / "export_dataset.csv").write_text("a\n1\n", encoding="utf-8")
    (report_dir / "report.html").write_text("<html></html>", encoding="utf-8")
    (audit_dir / "audit_report.json").write_text("{}", encoding="utf-8")
    controller = ArtifactController(
        artifact_dir=tmp_path / "artifacts",
        audit_dir=audit_dir,
        report_dir=report_dir,
        export_dir=export_dir,
    )

    assert [artifact.name for artifact in controller.list_exports()] == ["export_dataset.csv"]
    assert [artifact.name for artifact in controller.list_reports()] == ["report.html"]
    assert [artifact.name for artifact in controller.list_audit_files()] == ["audit_report.json"]


def test_artifact_controller_handles_missing_and_invalid_artifacts(tmp_path) -> None:
    controller = ArtifactController(export_dir=tmp_path / "exports")

    payload, warning = controller.read_json(tmp_path / "missing.json")
    assert payload is None
    assert warning and "Missing artifact" in warning

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("[1, 2]", encoding="utf-8")
    payload, warning = controller.read_json(invalid_path)
    assert payload is None
    assert warning and "must contain an object" in warning

    valid_path = tmp_path / "valid.json"
    valid_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    payload, warning = controller.read_json(valid_path)
    assert payload == {"ok": True}
    assert warning is None
