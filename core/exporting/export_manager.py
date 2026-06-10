from __future__ import annotations

from dataclasses import replace

from core.exporting.base_exporter import ExportContext
from core.exporting.export_registry import ExportRegistry
from models.export import ExportArtifact, ExportReport


class ExportManager:
    """Execute registered exporters in deterministic order."""

    export_order: tuple[str, ...] = ("csv", "xlsx", "json", "pdf")

    def __init__(self, registry: ExportRegistry) -> None:
        self.registry = registry

    def export(self, context: ExportContext) -> ExportReport:
        warnings = list(context.warnings)
        artifacts: list[ExportArtifact] = []

        for export_format in self.export_order:
            if export_format not in self.registry.available_exporters():
                warnings.append(f"Exporter not registered: {export_format}")
                continue
            exporter = self.registry.create(export_format)
            artifact = exporter.export(replace(context, warnings=warnings))
            artifacts.append(artifact)
            if artifact.warning is not None:
                warnings.append(artifact.warning)

        successful_exports = sum(1 for artifact in artifacts if artifact.exists)
        failed_exports = sum(1 for artifact in artifacts if not artifact.exists)
        return ExportReport(
            warnings=warnings,
            artifacts=artifacts,
            total_exports=len(artifacts),
            successful_exports=successful_exports,
            failed_exports=failed_exports,
        )
