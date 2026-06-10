from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from models.report import ReportChart


logger = logging.getLogger(__name__)


class PlotlyChartBuilder:
    """Build deterministic, domain-agnostic Plotly chart assets."""

    def __init__(self, max_numeric_charts: int = 5, max_categorical_charts: int = 5) -> None:
        self.max_numeric_charts = max_numeric_charts
        self.max_categorical_charts = max_categorical_charts

    def build_charts(
        self,
        dataframe: pd.DataFrame | None,
        semantic_payload: dict[str, Any] | None,
        rules_payload: dict[str, Any] | None,
        audit_payload: dict[str, Any] | None,
        output_dir: str | Path,
        warnings: list[str],
    ) -> list[ReportChart]:
        """Create chart assets and return metadata sorted by stable chart ID."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        charts: list[ReportChart] = []
        if dataframe is not None:
            charts.extend(self._build_numeric_histograms(dataframe, output_path, warnings))
            charts.extend(self._build_categorical_bars(dataframe, output_path, warnings))
        else:
            warnings.append("No dataframe artifact available for numeric or categorical charts.")

        semantic_chart = self._build_semantic_distribution(semantic_payload, output_path, warnings)
        if semantic_chart is not None:
            charts.append(semantic_chart)

        rule_chart = self._build_rule_distribution(rules_payload, output_path, warnings)
        if rule_chart is not None:
            charts.append(rule_chart)

        artifact_chart = self._build_artifact_sizes(audit_payload, output_path, warnings)
        if artifact_chart is not None:
            charts.append(artifact_chart)

        return sorted(charts, key=lambda chart: chart.chart_id)

    def _build_numeric_histograms(
        self,
        dataframe: pd.DataFrame,
        output_dir: Path,
        warnings: list[str],
    ) -> list[ReportChart]:
        charts: list[ReportChart] = []
        numeric_columns = [
            column
            for column in sorted(dataframe.columns)
            if is_numeric_dtype(dataframe[column]) and not is_bool_dtype(dataframe[column])
        ][: self.max_numeric_charts]

        for column in numeric_columns:
            series = dataframe[column].dropna()
            if series.empty:
                continue
            chart_id = f"numeric-{self._safe_id(column)}"
            filename = f"{chart_id}.html"
            figure = go.Figure(
                data=[
                    go.Histogram(
                        x=series,
                        marker_color="#2f6f73",
                    )
                ]
            )
            figure.update_layout(
                title=f"{column} Distribution",
                xaxis_title=column,
                yaxis_title="Rows",
                template="plotly_white",
            )
            self._write_chart(figure, output_dir / filename, chart_id)
            charts.append(
                ReportChart(
                    chart_id=chart_id,
                    title=f"{column} Distribution",
                    chart_type="histogram",
                    path=f"assets/{filename}",
                )
            )

        if not charts:
            warnings.append("No numeric columns available for histogram charts.")
        return charts

    def _build_categorical_bars(
        self,
        dataframe: pd.DataFrame,
        output_dir: Path,
        warnings: list[str],
    ) -> list[ReportChart]:
        charts: list[ReportChart] = []
        categorical_columns = [
            column
            for column in sorted(dataframe.columns)
            if not is_numeric_dtype(dataframe[column]) or is_bool_dtype(dataframe[column])
        ][: self.max_categorical_charts]

        for column in categorical_columns:
            counts = self._top_value_counts(dataframe[column], top_n=10)
            if not counts:
                continue
            values = [value for value, _count in counts]
            frequencies = [count for _value, count in counts]
            chart_id = f"categorical-{self._safe_id(column)}"
            filename = f"{chart_id}.html"
            figure = go.Figure(
                data=[
                    go.Bar(
                        x=values,
                        y=frequencies,
                        marker_color="#8a5a44",
                    )
                ]
            )
            figure.update_layout(
                title=f"{column} Top Values",
                xaxis_title=column,
                yaxis_title="Rows",
                template="plotly_white",
            )
            self._write_chart(figure, output_dir / filename, chart_id)
            charts.append(
                ReportChart(
                    chart_id=chart_id,
                    title=f"{column} Top Values",
                    chart_type="bar",
                    path=f"assets/{filename}",
                )
            )

        if not charts:
            warnings.append("No categorical columns available for top-value charts.")
        return charts

    def _build_semantic_distribution(
        self,
        semantic_payload: dict[str, Any] | None,
        output_dir: Path,
        warnings: list[str],
    ) -> ReportChart | None:
        columns = (semantic_payload or {}).get("columns", [])
        counts = Counter(str(column.get("semantic_type", "UNKNOWN")) for column in columns)
        if not counts:
            warnings.append("Semantic distribution chart skipped because semantic columns are missing.")
            return None

        labels = sorted(counts)
        values = [counts[label] for label in labels]
        chart_id = "semantic-distribution"
        filename = "semantic_distribution.html"
        figure = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.35)])
        figure.update_layout(title="Semantic Type Distribution", template="plotly_white")
        self._write_chart(figure, output_dir / filename, chart_id)
        return ReportChart(
            chart_id=chart_id,
            title="Semantic Type Distribution",
            chart_type="pie",
            path=f"assets/{filename}",
        )

    def _build_rule_distribution(
        self,
        rules_payload: dict[str, Any] | None,
        output_dir: Path,
        warnings: list[str],
    ) -> ReportChart | None:
        rules = (rules_payload or {}).get("rules", [])
        counts = Counter(str(rule.get("type", "unknown")) for rule in rules)
        if not counts:
            warnings.append("Rule distribution chart skipped because rules are missing.")
            return None

        rule_types = sorted(counts)
        values = [counts[rule_type] for rule_type in rule_types]
        chart_id = "rule-distribution"
        filename = "rule_distribution.html"
        figure = go.Figure(
            data=[
                go.Bar(
                    x=rule_types,
                    y=values,
                    marker_color="#5b6f9f",
                )
            ]
        )
        figure.update_layout(
            title="Rule Type Distribution",
            xaxis_title="Rule type",
            yaxis_title="Rules",
            template="plotly_white",
        )
        self._write_chart(figure, output_dir / filename, chart_id)
        return ReportChart(
            chart_id=chart_id,
            title="Rule Type Distribution",
            chart_type="bar",
            path=f"assets/{filename}",
        )

    def _build_artifact_sizes(
        self,
        audit_payload: dict[str, Any] | None,
        output_dir: Path,
        warnings: list[str],
    ) -> ReportChart | None:
        artifacts = [
            artifact
            for artifact in (audit_payload or {}).get("artifacts", [])
            if artifact.get("exists") and artifact.get("size_bytes") is not None
        ]
        if not artifacts:
            warnings.append("Artifact size chart skipped because audit artifact metadata is missing.")
            return None

        ordered_artifacts = sorted(artifacts, key=lambda artifact: str(artifact.get("artifact", "")))
        names = [str(artifact.get("artifact", "")) for artifact in ordered_artifacts]
        sizes = [int(artifact.get("size_bytes", 0)) for artifact in ordered_artifacts]
        chart_id = "artifact-sizes"
        filename = "artifact_sizes.html"
        figure = go.Figure(
            data=[
                go.Bar(
                    x=names,
                    y=sizes,
                    marker_color="#6c7a3f",
                )
            ]
        )
        figure.update_layout(
            title="Audit Artifact Sizes",
            xaxis_title="Artifact",
            yaxis_title="Bytes",
            template="plotly_white",
        )
        self._write_chart(figure, output_dir / filename, chart_id)
        return ReportChart(
            chart_id=chart_id,
            title="Audit Artifact Sizes",
            chart_type="bar",
            path=f"assets/{filename}",
        )

    def _top_value_counts(self, series: pd.Series, top_n: int) -> list[tuple[str, int]]:
        values = [str(value) for value in series.dropna().tolist() if str(value).strip()]
        counts = Counter(values)
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]

    def _write_chart(self, figure: go.Figure, output_path: Path, chart_id: str) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(
            output_path,
            include_plotlyjs=True,
            full_html=True,
            div_id=chart_id,
            config={"displayModeBar": False, "responsive": True},
        )
        logger.info("Wrote chart asset to %s", output_path)

    def _safe_id(self, value: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
        return normalized or "column"
