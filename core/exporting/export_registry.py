from __future__ import annotations

from core.exporting.base_exporter import BaseExporter


class ExportRegistry:
    """Explicit registry for pluggable exporter implementations."""

    def __init__(self) -> None:
        self._exporters: dict[str, type[BaseExporter]] = {}

    def register(self, export_format: str, exporter_class: type[BaseExporter]) -> None:
        self._exporters[export_format.strip().lower()] = exporter_class

    def create(self, export_format: str) -> BaseExporter:
        normalized_format = export_format.strip().lower()
        if normalized_format not in self._exporters:
            raise KeyError(f"Exporter not registered: {normalized_format}")
        return self._exporters[normalized_format]()

    def available_exporters(self) -> list[str]:
        return sorted(self._exporters)
