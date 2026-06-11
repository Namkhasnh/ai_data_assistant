from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from core.exporting.base_exporter import BaseExporter, ExportContext
from core.exporting.csv_exporter import CSVExporter
from core.exporting.export_registry import ExportRegistry
from core.exporting.json_exporter import JSONExporter
from core.exporting.xlsx_exporter import XLSXExporter
from models.export import ExportArtifact
from services.export_service import ExportService


class FailingPDFExporter(BaseExporter):
    format = "pdf"

    def export(self, context: ExportContext) -> ExportArtifact:
        output_path = context.export_dir / "pdf" / "export_report.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return ExportArtifact(
            artifact="export_report.pdf",
            format=self.format,
            path=str(output_path),
            exists=False,
            warning="PDF export unavailable",
        )


def test_export_service_generates_exports_report_and_preserves_upstream_artifacts(tmp_path):
    artifact_dir = tmp_path / "storage" / "artifacts"
    audit_dir = tmp_path / "storage" / "audit"
    report_dir = tmp_path / "storage" / "reports"
    export_dir = tmp_path / "storage" / "exports"
    artifact_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "b": [1],
            "a": ["enriched"],
            "standardized_title": ["Data Analyst"],
            "standardized_location": ["Hà Nội"],
            "job_family": ["Analytics"],
            "job_domain": ["Data"],
            "region": ["North"],
            "country": ["Vietnam"],
        }
    ).to_csv(
        artifact_dir / "enriched_dataset.csv",
        index=False,
    )
    pd.DataFrame({"b": [2], "a": ["standardized"]}).to_csv(
        artifact_dir / "standardized_dataset.csv",
        index=False,
    )
    _write_json(
        audit_dir / "audit_report.json",
        {
            "generated_at": "2026-06-10T00:00:00Z",
            "warnings": [],
            "artifacts": [
                {
                    "artifact": "metadata.json",
                    "module": "profiling",
                    "exists": True,
                    "size_bytes": 10,
                }
            ],
        },
    )
    (report_dir / "report.html").write_text("<html><body>Report</body></html>", encoding="utf-8")
    upstream_files = [
        artifact_dir / "enriched_dataset.csv",
        artifact_dir / "standardized_dataset.csv",
        audit_dir / "audit_report.json",
        report_dir / "report.html",
    ]
    before_hashes = {path: _sha256(path) for path in upstream_files}

    report = ExportService(
        artifact_dir=artifact_dir,
        audit_dir=audit_dir,
        report_dir=report_dir,
        export_dir=export_dir,
        registry=_registry_with_failing_pdf(),
    ).export_all()

    assert (export_dir / "csv" / "export_dataset.csv").exists()
    assert (export_dir / "xlsx" / "export_dataset.xlsx").exists()
    assert (export_dir / "json" / "export_dataset.json").exists()
    assert (export_dir / "export_report.json").exists()
    csv_lines = (export_dir / "csv" / "export_dataset.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert csv_lines[0] == (
        "b,a,standardized_title,standardized_location,"
        "job_family,job_domain,region,country"
    )
    assert csv_lines[1] == "1,enriched,Data Analyst,Hà Nội,Analytics,Data,North,Vietnam"
    assert [artifact.format for artifact in report.artifacts] == ["csv", "xlsx", "json", "pdf"]
    assert report.total_exports == 4
    assert report.successful_exports == 3
    assert report.failed_exports == 1
    assert report.artifacts[-1].warning == "PDF export unavailable"
    assert before_hashes == {path: _sha256(path) for path in upstream_files}

    persisted = json.loads((export_dir / "export_report.json").read_text(encoding="utf-8"))
    assert persisted["total_exports"] == 4
    assert persisted["successful_exports"] == 3
    assert persisted["failed_exports"] == 1
    assert persisted["generated_at"].endswith("Z")


def test_export_service_warns_when_upstream_artifacts_are_missing(tmp_path):
    report = ExportService(
        artifact_dir=tmp_path / "missing_artifacts",
        audit_dir=tmp_path / "missing_audit",
        report_dir=tmp_path / "missing_reports",
        export_dir=tmp_path / "exports",
        registry=_registry_with_failing_pdf(),
    ).export_all()

    assert report.total_exports == 4
    assert report.successful_exports == 0
    assert report.failed_exports == 4
    assert any("semantic_dataset.csv not found" in warning for warning in report.warnings)
    assert any("enriched_dataset.csv not found" in warning for warning in report.warnings)
    assert any("standardized_dataset.csv not found" in warning for warning in report.warnings)
    assert any("audit_report.json not found" in warning for warning in report.warnings)
    assert any("report.html not found" in warning for warning in report.warnings)
    assert report.warnings[-1] == "PDF export unavailable"
    assert (tmp_path / "exports" / "export_report.json").exists()


def _registry_with_failing_pdf() -> ExportRegistry:
    registry = ExportRegistry()
    registry.register("csv", CSVExporter)
    registry.register("xlsx", XLSXExporter)
    registry.register("json", JSONExporter)
    registry.register("pdf", FailingPDFExporter)
    return registry


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
