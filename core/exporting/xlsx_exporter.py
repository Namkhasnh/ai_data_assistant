from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.exporting.base_exporter import BaseExporter, ExportContext
from models.export import ExportArtifact


class XLSXExporter(BaseExporter):
    """Export the selected dataset and audit context as a deterministic workbook."""

    format = "xlsx"

    def export(self, context: ExportContext) -> ExportArtifact:
        output_path = Path(context.export_dir) / "xlsx" / "export_dataset.xlsx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if context.dataframe is None:
            return self._artifact(
                artifact="export_dataset.xlsx",
                output_path=output_path,
                warning="Dataset export unavailable",
            )

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            context.dataframe.copy(deep=True).to_excel(
                writer,
                sheet_name="Dataset",
                index=False,
            )
            self._audit_summary_frame(context.audit_report).to_excel(
                writer,
                sheet_name="Audit Summary",
                index=False,
            )
            self._warnings_frame(context.warnings).to_excel(
                writer,
                sheet_name="Warnings",
                index=False,
            )
        return self._artifact(artifact="export_dataset.xlsx", output_path=output_path)

    def _audit_summary_frame(self, audit_report: dict[str, Any] | None) -> pd.DataFrame:
        artifacts = (audit_report or {}).get("artifacts", [])
        if not artifacts:
            return pd.DataFrame(
                [{"artifact": None, "module": None, "exists": None, "size_bytes": None}]
            )
        rows = [
            {
                "artifact": artifact.get("artifact"),
                "module": artifact.get("module"),
                "exists": artifact.get("exists"),
                "size_bytes": artifact.get("size_bytes"),
            }
            for artifact in sorted(artifacts, key=lambda item: str(item.get("artifact", "")))
        ]
        return pd.DataFrame(rows)

    def _warnings_frame(self, warnings: list[str]) -> pd.DataFrame:
        if not warnings:
            return pd.DataFrame([{"warning": None}])
        return pd.DataFrame([{"warning": warning} for warning in warnings])
