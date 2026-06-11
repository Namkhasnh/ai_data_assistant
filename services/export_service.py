from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from core.exporting.base_exporter import ExportContext
from core.exporting.csv_exporter import CSVExporter
from core.exporting.export_manager import ExportManager
from core.exporting.export_registry import ExportRegistry
from core.exporting.json_exporter import JSONExporter
from core.exporting.pdf_exporter import PDFExporter
from core.exporting.xlsx_exporter import XLSXExporter
from models.export import ExportReport


logger = logging.getLogger(__name__)


class ExportService:
    """Thin orchestration service for passive export generation."""

    def __init__(
        self,
        artifact_dir: str | Path = "storage/artifacts",
        audit_dir: str | Path = "storage/audit",
        report_dir: str | Path = "storage/reports",
        export_dir: str | Path = "storage/exports",
        registry: ExportRegistry | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.audit_dir = Path(audit_dir)
        self.report_dir = Path(report_dir)
        self.export_dir = Path(export_dir)
        self.registry = registry or self._default_registry()
        self.manager = ExportManager(registry=self.registry)

    def export_all(self) -> ExportReport:
        """Export available artifacts and persist export_report.json."""

        warnings: list[str] = []
        dataframe = self._load_dataset(warnings)
        audit_report = self._load_json(self.audit_dir / "audit_report.json", warnings)
        report_html_path = self.report_dir / "report.html"
        if not report_html_path.exists():
            warnings.append(f"report.html not found: {report_html_path}")

        context = ExportContext(
            dataframe=dataframe,
            audit_report=audit_report,
            report_html_path=report_html_path,
            export_dir=self.export_dir,
            warnings=warnings,
        )
        report = self.manager.export(context)
        self._write_report(report)
        return report

    def _default_registry(self) -> ExportRegistry:
        registry = ExportRegistry()
        registry.register("csv", CSVExporter)
        registry.register("xlsx", XLSXExporter)
        registry.register("json", JSONExporter)
        registry.register("pdf", PDFExporter)
        return registry

    def _load_dataset(self, warnings: list[str]) -> pd.DataFrame | None:
        semantic_path = self.artifact_dir / "semantic_dataset.csv"
        business_path = self.artifact_dir / "business_dataset.csv"
        enriched_path = self.artifact_dir / "enriched_dataset.csv"
        standardized_path = self.artifact_dir / "standardized_dataset.csv"

        if semantic_path.exists():
            dataframe = self._read_csv(semantic_path, warnings)
            if dataframe is not None:
                return dataframe
        else:
            warnings.append(f"semantic_dataset.csv not found: {semantic_path}")

        if business_path.exists():
            dataframe = self._read_csv(business_path, warnings)
            if dataframe is not None:
                return dataframe
        else:
            warnings.append(f"business_dataset.csv not found: {business_path}")

        if enriched_path.exists():
            dataframe = self._read_csv(enriched_path, warnings)
            if dataframe is not None:
                return dataframe
        else:
            warnings.append(f"enriched_dataset.csv not found: {enriched_path}")

        if standardized_path.exists():
            dataframe = self._read_csv(standardized_path, warnings)
            if dataframe is not None:
                return dataframe
        else:
            warnings.append(f"standardized_dataset.csv not found: {standardized_path}")

        return None

    def _read_csv(self, path: Path, warnings: list[str]) -> pd.DataFrame | None:
        try:
            return pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 - export must degrade gracefully.
            warnings.append(f"Unable to read dataset artifact {path}: {exc}")
            return None

    def _load_json(self, path: Path, warnings: list[str]) -> dict[str, Any] | None:
        if not path.exists():
            warnings.append(f"audit_report.json not found: {path}")
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            warnings.append(f"Invalid JSON artifact {path}: {exc}")
            return None

    def _write_report(self, report: ExportReport) -> Path:
        output_path = self.export_dir / "export_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote export report to %s", output_path)
        return output_path
