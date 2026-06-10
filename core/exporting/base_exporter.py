from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from models.export import ExportArtifact


@dataclass(frozen=True)
class ExportContext:
    """Read-only inputs and output location for export execution."""

    dataframe: pd.DataFrame | None
    audit_report: dict[str, Any] | None
    report_html_path: Path | None
    export_dir: Path
    warnings: list[str]


class BaseExporter(ABC):
    """Base interface implemented by all passive exporters."""

    format: str

    @abstractmethod
    def export(self, context: ExportContext) -> ExportArtifact:
        """Export one artifact and return its metadata."""

    def _artifact(
        self,
        artifact: str,
        output_path: Path,
        warning: str | None = None,
    ) -> ExportArtifact:
        exists = output_path.exists()
        return ExportArtifact(
            artifact=artifact,
            format=self.format,
            path=str(output_path),
            exists=exists,
            size_bytes=output_path.stat().st_size if exists else None,
            warning=warning,
        )
