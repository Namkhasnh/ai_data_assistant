from __future__ import annotations

import logging
from pathlib import Path

from core.exporting.base_exporter import BaseExporter, ExportContext
from models.export import ExportArtifact


logger = logging.getLogger(__name__)


class PDFExporter(BaseExporter):
    """Best-effort HTML-to-PDF exporter for the generated report."""

    format = "pdf"
    unavailable_warning = "PDF export unavailable"

    def export(self, context: ExportContext) -> ExportArtifact:
        output_path = Path(context.export_dir) / "pdf" / "export_report.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if context.report_html_path is None or not context.report_html_path.exists():
            self._remove_stale_output(output_path)
            return self._artifact(
                artifact="export_report.pdf",
                output_path=output_path,
                warning=self.unavailable_warning,
            )

        try:
            from weasyprint import HTML  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001 - optional dependency must be non-fatal.
            logger.info("PDF export dependency unavailable: %s", exc)
            self._remove_stale_output(output_path)
            return self._artifact(
                artifact="export_report.pdf",
                output_path=output_path,
                warning=self.unavailable_warning,
            )

        try:
            HTML(filename=str(context.report_html_path)).write_pdf(str(output_path))
        except Exception as exc:  # noqa: BLE001 - PDF export is best-effort.
            logger.info("PDF export failed: %s", exc)
            self._remove_stale_output(output_path)
            return self._artifact(
                artifact="export_report.pdf",
                output_path=output_path,
                warning=self.unavailable_warning,
            )

        return self._artifact(artifact="export_report.pdf", output_path=output_path)

    def _remove_stale_output(self, output_path: Path) -> None:
        if output_path.exists():
            output_path.unlink()
