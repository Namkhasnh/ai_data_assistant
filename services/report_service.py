from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from core.reporting.plotly_chart_builder import PlotlyChartBuilder
from core.reporting.report_generator import ReportGenerator
from models.dataset import DatasetMetadata
from models.report import PipelineReport, ReportMetadata, ReportSection
from models.semantic_tag import SemanticDetectionReport


logger = logging.getLogger(__name__)


class ReportService:
    """Read existing pipeline artifacts and produce a passive HTML report."""

    def __init__(
        self,
        artifact_dir: str | Path = "storage/artifacts",
        audit_dir: str | Path = "storage/audit",
        rules_path: str | Path = "data/rules/rules.json",
        report_dir: str | Path = "storage/reports",
        generator: ReportGenerator | None = None,
        chart_builder: PlotlyChartBuilder | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.audit_dir = Path(audit_dir)
        self.rules_path = Path(rules_path)
        self.report_dir = Path(report_dir)
        self.generator = generator or ReportGenerator()
        self.chart_builder = chart_builder or PlotlyChartBuilder()

    def generate_report(self) -> tuple[PipelineReport, Path]:
        """Generate report.html from existing artifacts without mutating inputs."""

        warnings: list[str] = []
        metadata_payload = self._load_json(self.artifact_dir / "metadata.json", warnings)
        semantic_payload = self._load_json(self.artifact_dir / "semantic_columns.json", warnings)
        rules_payload = self._load_rules(warnings)
        execution_payload = self._load_json(
            self.artifact_dir / "rule_execution_report.json", warnings
        )
        standardization_payload = self._load_json(
            self.artifact_dir / "standardization_report.json", warnings
        )
        enrichment_payload = self._load_json(self.artifact_dir / "enrichment_report.json", warnings)
        ai_payload = self._load_json(self.artifact_dir / "ai_suggestions.json", warnings)
        audit_payload = self._load_json(self.audit_dir / "audit_report.json", warnings)
        dataframe = self._load_result_dataframe(warnings)

        metadata = self._parse_metadata(metadata_payload, warnings)
        semantic_report = self._parse_semantic_report(semantic_payload, warnings)

        sections = [
            self._dataset_summary(metadata, metadata_payload),
            self._column_analysis(metadata, semantic_report),
            self._semantic_summary(semantic_report),
            self._rule_summary(rules_payload, execution_payload),
            self._standardization_summary(standardization_payload, dataframe),
            self._enrichment_summary(enrichment_payload, dataframe, metadata),
            self._audit_summary(audit_payload),
            self._before_after_summary(dataframe, metadata),
            self._ai_summary(ai_payload),
        ]
        report = PipelineReport(
            metadata=ReportMetadata(
                title="Pipeline Report",
                source_file=metadata.source_file if metadata is not None else None,
            ),
            sections=sections,
            warnings=warnings,
        )
        charts = self.chart_builder.build_charts(
            dataframe=dataframe,
            semantic_payload=semantic_payload,
            rules_payload=rules_payload,
            audit_payload=audit_payload,
            output_dir=self.report_dir / "assets",
            warnings=report.warnings,
        )
        output_path = self.generator.render(
            report=report,
            charts=charts,
            output_path=self.report_dir / "report.html",
        )
        return report, output_path

    def _load_json(self, path: Path, warnings: list[str]) -> dict[str, Any] | None:
        if not path.exists():
            warnings.append(f"Missing artifact: {path}")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.append(f"Invalid JSON artifact {path}: {exc}")
            return None

    def _load_rules(self, warnings: list[str]) -> dict[str, Any] | None:
        if self.rules_path.exists():
            return self._load_json(self.rules_path, warnings)
        fallback_path = self.artifact_dir / "rules.json"
        return self._load_json(fallback_path, warnings)

    def _load_result_dataframe(self, warnings: list[str]) -> pd.DataFrame | None:
        candidates = [
            self.artifact_dir / "enriched_dataset.csv",
            self.artifact_dir / "standardized_dataset.csv",
            self.artifact_dir / "cleaned_dataset.csv",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                return pd.read_csv(path)
            except Exception as exc:  # noqa: BLE001 - report generation must degrade gracefully.
                warnings.append(f"Unable to read dataframe artifact {path}: {exc}")
                return None
        warnings.append("Missing dataframe artifact: enriched_dataset.csv, standardized_dataset.csv, cleaned_dataset.csv")
        return None

    def _parse_metadata(
        self,
        payload: dict[str, Any] | None,
        warnings: list[str],
    ) -> DatasetMetadata | None:
        if payload is None:
            return None
        try:
            return DatasetMetadata.model_validate(payload)
        except ValidationError as exc:
            warnings.append(f"metadata.json did not match DatasetMetadata schema: {exc}")
            return None

    def _parse_semantic_report(
        self,
        payload: dict[str, Any] | None,
        warnings: list[str],
    ) -> SemanticDetectionReport | None:
        if payload is None:
            return None
        try:
            return SemanticDetectionReport.model_validate(payload)
        except ValidationError as exc:
            warnings.append(f"semantic_columns.json did not match SemanticDetectionReport schema: {exc}")
            return None

    def _dataset_summary(
        self,
        metadata: DatasetMetadata | None,
        payload: dict[str, Any] | None,
    ) -> ReportSection:
        if metadata is None:
            return ReportSection(
                section_id="dataset_summary",
                title="Dataset Summary",
                summary="Dataset metadata is unavailable.",
                data={},
            )
        missing_values = sum(column.null_count for column in metadata.columns)
        return ReportSection(
            section_id="dataset_summary",
            title="Dataset Summary",
            summary="Profiled dataset shape and missingness.",
            data={
                "source_file": metadata.source_file,
                "file_format": metadata.file_format,
                "row_count": metadata.row_count,
                "column_count": metadata.column_count,
                "duplicate_count": metadata.duplicate_count,
                "missing_values": missing_values,
                "raw": payload or {},
            },
        )

    def _column_analysis(
        self,
        metadata: DatasetMetadata | None,
        semantic_report: SemanticDetectionReport | None,
    ) -> ReportSection:
        semantic_by_column = {
            tag.column_name: tag for tag in semantic_report.columns
        } if semantic_report is not None else {}
        rows: list[dict[str, Any]] = []
        if metadata is not None:
            for column in sorted(metadata.columns, key=lambda profile: profile.name):
                semantic = semantic_by_column.get(column.name)
                rows.append(
                    {
                        "name": column.name,
                        "data_type": column.data_type,
                        "null_count": column.null_count,
                        "null_percentage": column.null_percentage,
                        "unique_value_count": column.unique_value_count,
                        "top_values": [
                            f"{self._compact_value(top.value)} ({top.count})"
                            for top in column.top_values[:3]
                        ],
                        "semantic_type": semantic.semantic_type if semantic is not None else "UNKNOWN",
                        "semantic_confidence": semantic.confidence if semantic is not None else 0.0,
                    }
                )
        return ReportSection(
            section_id="column_analysis",
            title="Column Analysis",
            summary="Column profiling metadata combined with semantic labels.",
            data={"columns": rows},
        )

    def _semantic_summary(
        self,
        semantic_report: SemanticDetectionReport | None,
    ) -> ReportSection:
        columns = []
        if semantic_report is not None:
            columns = [
                {
                    "column_name": tag.column_name,
                    "semantic_type": tag.semantic_type,
                    "confidence": tag.confidence,
                    "detector_name": tag.detector_name,
                }
                for tag in sorted(semantic_report.columns, key=lambda tag: tag.column_name)
            ]
        return ReportSection(
            section_id="semantic_summary",
            title="Semantic Summary",
            summary="Semantic detections generated by deterministic detectors.",
            data={"columns": columns},
        )

    def _rule_summary(
        self,
        rules_payload: dict[str, Any] | None,
        execution_payload: dict[str, Any] | None,
    ) -> ReportSection:
        rules = (rules_payload or {}).get("rules", [])
        results = (execution_payload or {}).get("results", [])
        counts_by_type = dict(sorted(Counter(str(rule.get("type", "unknown")) for rule in rules).items()))
        status_counts = dict(sorted(Counter(str(result.get("status", "unknown")) for result in results).items()))
        return ReportSection(
            section_id="rule_summary",
            title="Rule Summary",
            summary="Rule inventory and execution outcomes.",
            data={
                "total_rules": len(rules),
                "counts_by_type": counts_by_type,
                "execution_status_counts": status_counts,
                "total_affected_rows": sum(int(result.get("affected_rows", 0)) for result in results),
                "rules": sorted(
                    [
                        {
                            "id": rule.get("id", ""),
                            "type": rule.get("type", ""),
                            "column": rule.get("column", ""),
                            "enabled": rule.get("enabled", False),
                            "priority": rule.get("priority", 100),
                        }
                        for rule in rules
                    ],
                    key=lambda rule: (str(rule["type"]), str(rule["id"])),
                ),
            },
        )

    def _standardization_summary(
        self,
        standardization_payload: dict[str, Any] | None,
        dataframe: pd.DataFrame | None,
    ) -> ReportSection:
        return ReportSection(
            section_id="standardization_summary",
            title="Standardization Summary",
            summary="Deterministic standardization output metadata.",
            data={
                "total_standardized_columns": (standardization_payload or {}).get(
                    "total_standardized_columns", 0
                ),
                "standardized_by_standardizer": (standardization_payload or {}).get(
                    "standardized_by_standardizer", {}
                ),
                "warnings": (standardization_payload or {}).get("warnings", []),
                "skipped_columns": (standardization_payload or {}).get("skipped_columns", []),
                "examples": self._standardization_examples(dataframe),
            },
        )

    def _enrichment_summary(
        self,
        enrichment_payload: dict[str, Any] | None,
        dataframe: pd.DataFrame | None,
        metadata: DatasetMetadata | None,
    ) -> ReportSection:
        return ReportSection(
            section_id="enrichment_summary",
            title="Enrichment Summary",
            summary="Deterministic enrichment output metadata.",
            data={
                "total_enriched_columns": (enrichment_payload or {}).get("total_enriched_columns", 0),
                "enriched_by_enricher": (enrichment_payload or {}).get("enriched_by_enricher", {}),
                "warnings": (enrichment_payload or {}).get("warnings", []),
                "skipped_columns": (enrichment_payload or {}).get("skipped_columns", []),
                "examples": self._enrichment_examples(dataframe, metadata),
            },
        )

    def _audit_summary(self, audit_payload: dict[str, Any] | None) -> ReportSection:
        artifacts = sorted(
            (audit_payload or {}).get("artifacts", []),
            key=lambda artifact: str(artifact.get("artifact", "")),
        )
        rule_history = sorted(
            (audit_payload or {}).get("rule_history", {}).get("records", []),
            key=lambda record: str(record.get("rule_id", "")),
        )
        return ReportSection(
            section_id="audit_summary",
            title="Audit Summary",
            summary="Passive audit metadata assembled from existing artifacts.",
            data={
                "warnings": (audit_payload or {}).get("warnings", []),
                "artifacts": artifacts,
                "rule_history": rule_history,
            },
        )

    def _before_after_summary(
        self,
        dataframe: pd.DataFrame | None,
        metadata: DatasetMetadata | None,
    ) -> ReportSection:
        return ReportSection(
            section_id="before_after",
            title="Before And After",
            summary="Representative raw, standardized, and enriched values.",
            data={"examples": self._before_after_examples(dataframe, metadata)},
        )

    def _ai_summary(self, ai_payload: dict[str, Any] | None) -> ReportSection:
        suggestions = (ai_payload or {}).get("suggestions", [])
        return ReportSection(
            section_id="ai_summary",
            title="AI Suggestions",
            summary="Suggestion-only local AI output pending human review.",
            data={
                "suggestion_count": len(suggestions),
                "warnings": (ai_payload or {}).get("warnings", []),
                "provider": (ai_payload or {}).get("provider"),
                "model": (ai_payload or {}).get("model"),
            },
        )

    def _standardization_examples(self, dataframe: pd.DataFrame | None) -> list[dict[str, Any]]:
        if dataframe is None:
            return []
        examples: list[dict[str, Any]] = []
        for standardized_column in sorted(
            column for column in dataframe.columns if column.startswith("standardized_")
        ):
            source_column = standardized_column.removeprefix("standardized_")
            if source_column not in dataframe.columns:
                continue
            for _index, row in dataframe[[source_column, standardized_column]].iterrows():
                raw_value = row[source_column]
                standardized_value = row[standardized_column]
                if self._is_unknown(raw_value) or self._is_unknown(standardized_value):
                    continue
                examples.append(
                    {
                        "source_column": source_column,
                        "source_value": raw_value,
                        "standardized_column": standardized_column,
                        "standardized_value": standardized_value,
                    }
                )
                break
        return examples[:10]

    def _enrichment_examples(
        self,
        dataframe: pd.DataFrame | None,
        metadata: DatasetMetadata | None,
    ) -> list[dict[str, Any]]:
        if dataframe is None:
            return []
        source_columns = self._source_columns(dataframe, metadata)
        ignored_prefixes = ("standardized_", "validation_")
        derived_columns = [
            column
            for column in sorted(dataframe.columns)
            if column not in source_columns
            and not column.startswith(ignored_prefixes)
            and dataframe[column].notna().any()
        ]
        examples: list[dict[str, Any]] = []
        for column in derived_columns:
            first_valid = dataframe[column].dropna()
            if first_valid.empty:
                continue
            value = first_valid.iloc[0]
            if self._is_unknown(value):
                continue
            examples.append({"column": column, "value": value})
        return examples[:10]

    def _before_after_examples(
        self,
        dataframe: pd.DataFrame | None,
        metadata: DatasetMetadata | None,
    ) -> list[dict[str, Any]]:
        if dataframe is None:
            return []
        source_columns = self._source_columns(dataframe, metadata)
        enrichment_columns = [
            column
            for column in sorted(dataframe.columns)
            if not column.startswith("standardized_")
            and column not in source_columns
            and not column.startswith("validation_")
        ]
        examples: list[dict[str, Any]] = []
        for standardized_column in sorted(
            column for column in dataframe.columns if column.startswith("standardized_")
        ):
            source_column = standardized_column.removeprefix("standardized_")
            if source_column not in dataframe.columns:
                continue
            selected_columns = [source_column, standardized_column] + enrichment_columns
            for _index, row in dataframe[selected_columns].iterrows():
                raw_value = row[source_column]
                standardized_value = row[standardized_column]
                if self._is_unknown(raw_value) or self._is_unknown(standardized_value):
                    continue
                enrichments = {
                    column: row[column]
                    for column in enrichment_columns
                    if not self._is_unknown(row[column])
                }
                examples.append(
                    {
                        "source_column": source_column,
                        "raw_value": raw_value,
                        "standardized_column": standardized_column,
                        "standardized_value": standardized_value,
                        "enrichments": enrichments,
                    }
                )
                break
        return examples[:10]

    def _source_columns(
        self,
        dataframe: pd.DataFrame,
        metadata: DatasetMetadata | None,
    ) -> set[str]:
        if metadata is not None:
            return {column.name for column in metadata.columns}
        standardized_sources = {
            column.removeprefix("standardized_")
            for column in dataframe.columns
            if column.startswith("standardized_")
        }
        return {column for column in dataframe.columns if column in standardized_sources}

    def _is_unknown(self, value: Any) -> bool:
        if pd.isna(value):
            return True
        normalized = str(value).strip().lower()
        return normalized in {"", "unknown", "none", "nan", "null", "n/a"}

    def _compact_value(self, value: Any, max_length: int = 120) -> str:
        text = str(value).replace("\n", " ").strip()
        if len(text) <= max_length:
            return text
        return f"{text[: max_length - 3]}..."
