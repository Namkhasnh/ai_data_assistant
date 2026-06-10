from __future__ import annotations

from pathlib import Path

from core.exporting.base_exporter import BaseExporter, ExportContext
from models.export import ExportArtifact


class CSVExporter(BaseExporter):
    """Export the selected dataset as CSV while preserving column order."""

    format = "csv"

    def export(self, context: ExportContext) -> ExportArtifact:
        output_path = Path(context.export_dir) / "csv" / "export_dataset.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if context.dataframe is None:
            return self._artifact(
                artifact="export_dataset.csv",
                output_path=output_path,
                warning="Dataset export unavailable",
            )

        context.dataframe.copy(deep=True).to_csv(output_path, index=False)
        return self._artifact(artifact="export_dataset.csv", output_path=output_path)
