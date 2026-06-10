from __future__ import annotations

from core.reporting.report_generator import ReportGenerator
from models.report import PipelineReport, ReportChart, ReportMetadata, ReportSection


def test_report_generator_renders_html_with_external_chart_references(tmp_path):
    report = PipelineReport(
        metadata=ReportMetadata(source_file="sample.csv"),
        warnings=["Missing artifact: optional.json"],
        sections=[
            ReportSection(
                section_id="dataset_summary",
                title="Dataset Summary",
                data={
                    "row_count": 2,
                    "column_count": 2,
                    "duplicate_count": 0,
                    "missing_values": 0,
                },
            ),
            ReportSection(
                section_id="column_analysis",
                title="Column Analysis",
                data={"columns": []},
            ),
        ],
    )
    charts = [
        ReportChart(
            chart_id="semantic-distribution",
            title="Semantic Type Distribution",
            chart_type="pie",
            path="assets/semantic_distribution.html",
        )
    ]

    output_path = ReportGenerator().render(
        report=report,
        charts=charts,
        output_path=tmp_path / "report.html",
    )

    html = output_path.read_text(encoding="utf-8")
    assert "Pipeline Report" in html
    assert "Missing artifact: optional.json" in html
    assert "Dataset Summary" in html
    assert "assets/semantic_distribution.html" in html
    assert report.metadata.model_dump(mode="json")["generated_at"].endswith("Z")
