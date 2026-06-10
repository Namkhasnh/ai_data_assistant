from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from core.exporting.base_exporter import BaseExporter, ExportContext
from models.export import ExportArtifact


class JSONExporter(BaseExporter):
    """Export the selected dataset as deterministic records-oriented JSON."""

    format = "json"

    def export(self, context: ExportContext) -> ExportArtifact:
        output_path = Path(context.export_dir) / "json" / "export_dataset.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if context.dataframe is None:
            return self._artifact(
                artifact="export_dataset.json",
                output_path=output_path,
                warning="Dataset export unavailable",
            )

        records = self._records(context.dataframe)
        output_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return self._artifact(artifact="export_dataset.json", output_path=output_path)

    def _records(self, dataframe: pd.DataFrame) -> list[dict[str, Any]]:
        sanitized = dataframe.copy(deep=True).astype(object).where(pd.notnull(dataframe), None)
        return sanitized.to_dict(orient="records")
