from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from models.report import PipelineReport, ReportChart, ReportSection


logger = logging.getLogger(__name__)


class ReportGenerator:
    """Render a PipelineReport into HTML without owning data assembly."""

    def __init__(self, template_dir: str | Path = "core/reporting/templates") -> None:
        self.template_dir = Path(template_dir)
        self.environment = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(
        self,
        report: PipelineReport,
        charts: list[ReportChart],
        output_path: str | Path = "storage/reports/report.html",
    ) -> Path:
        """Render report.html and return its path."""

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        template = self.environment.get_template("base.html")
        html = template.render(
            report=report,
            metadata=report.metadata.model_dump(mode="json"),
            sections_by_id=self._sections_by_id(report.sections),
            charts=sorted(charts, key=lambda chart: chart.chart_id),
        )
        output.write_text(html, encoding="utf-8")
        logger.info("Wrote HTML report to %s", output)
        return output

    def _sections_by_id(self, sections: list[ReportSection]) -> dict[str, ReportSection]:
        return {section.section_id: section for section in sections}
