from __future__ import annotations

from core.exporting.base_exporter import BaseExporter, ExportContext
from core.exporting.export_manager import ExportManager
from core.exporting.export_registry import ExportRegistry
from models.export import ExportArtifact


def test_export_manager_executes_exporters_in_deterministic_order(tmp_path):
    calls: list[str] = []
    registry = ExportRegistry()
    registry.register("pdf", _exporter("pdf", calls))
    registry.register("json", _exporter("json", calls))
    registry.register("xlsx", _exporter("xlsx", calls))
    registry.register("csv", _exporter("csv", calls))
    context = ExportContext(
        dataframe=None,
        audit_report=None,
        report_html_path=None,
        export_dir=tmp_path,
        warnings=[],
    )

    report = ExportManager(registry=registry).export(context)

    assert calls == ["csv", "xlsx", "json", "pdf"]
    assert [artifact.format for artifact in report.artifacts] == ["csv", "xlsx", "json", "pdf"]
    assert report.total_exports == 4
    assert report.successful_exports == 4
    assert report.failed_exports == 0


def test_export_manager_keeps_pdf_failure_non_fatal(tmp_path):
    calls: list[str] = []
    registry = ExportRegistry()
    registry.register("csv", _exporter("csv", calls))
    registry.register("xlsx", _exporter("xlsx", calls))
    registry.register("json", _exporter("json", calls))
    registry.register("pdf", _exporter("pdf", calls, exists=False, warning="PDF export unavailable"))
    context = ExportContext(
        dataframe=None,
        audit_report=None,
        report_html_path=None,
        export_dir=tmp_path,
        warnings=[],
    )

    report = ExportManager(registry=registry).export(context)

    assert calls == ["csv", "xlsx", "json", "pdf"]
    assert report.total_exports == 4
    assert report.successful_exports == 3
    assert report.failed_exports == 1
    assert report.warnings == ["PDF export unavailable"]


def _exporter(
    export_format: str,
    calls: list[str],
    exists: bool = True,
    warning: str | None = None,
) -> type[BaseExporter]:
    class RecordingExporter(BaseExporter):
        format = export_format

        def export(self, context: ExportContext) -> ExportArtifact:
            calls.append(export_format)
            return ExportArtifact(
                artifact=f"{export_format}.out",
                format=export_format,
                path=str(context.export_dir / f"{export_format}.out"),
                exists=exists,
                size_bytes=1 if exists else None,
                warning=warning,
            )

    return RecordingExporter
