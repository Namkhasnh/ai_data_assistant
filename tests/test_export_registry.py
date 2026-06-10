from __future__ import annotations

import pytest

from core.exporting.base_exporter import BaseExporter, ExportContext
from core.exporting.export_registry import ExportRegistry
from models.export import ExportArtifact


class DummyExporter(BaseExporter):
    format = "dummy"

    def export(self, context: ExportContext) -> ExportArtifact:
        return ExportArtifact(
            artifact="dummy.txt",
            format=self.format,
            path=str(context.export_dir / "dummy.txt"),
            exists=False,
        )


def test_export_registry_registers_and_creates_exporters():
    registry = ExportRegistry()

    registry.register("dummy", DummyExporter)

    assert registry.available_exporters() == ["dummy"]
    assert isinstance(registry.create("dummy"), DummyExporter)


def test_export_registry_rejects_unknown_exporter():
    registry = ExportRegistry()

    with pytest.raises(KeyError):
        registry.create("missing")
