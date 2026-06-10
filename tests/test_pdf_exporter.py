from __future__ import annotations

import builtins

from core.exporting.base_exporter import ExportContext
from core.exporting.pdf_exporter import PDFExporter


def test_pdf_exporter_returns_warning_when_dependency_is_unavailable(tmp_path, monkeypatch):
    report_path = tmp_path / "report.html"
    report_path.write_text("<html><body>Report</body></html>", encoding="utf-8")
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "weasyprint":
            raise ImportError("weasyprint unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    context = ExportContext(
        dataframe=None,
        audit_report=None,
        report_html_path=report_path,
        export_dir=tmp_path / "exports",
        warnings=[],
    )

    artifact = PDFExporter().export(context)

    assert artifact.exists is False
    assert artifact.warning == "PDF export unavailable"
    assert artifact.size_bytes is None


def test_pdf_exporter_handles_missing_report_without_exception(tmp_path):
    context = ExportContext(
        dataframe=None,
        audit_report=None,
        report_html_path=tmp_path / "missing.html",
        export_dir=tmp_path / "exports",
        warnings=[],
    )

    artifact = PDFExporter().export(context)

    assert artifact.exists is False
    assert artifact.warning == "PDF export unavailable"
    assert (tmp_path / "exports" / "pdf").exists()
