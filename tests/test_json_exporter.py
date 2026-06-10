from __future__ import annotations

import pandas as pd

from core.exporting.base_exporter import ExportContext
from core.exporting.json_exporter import JSONExporter


def test_json_exporter_uses_records_orientation_and_deterministic_formatting(tmp_path):
    dataframe = pd.DataFrame({"b": [1, None], "a": ["x", "y"]})
    context = ExportContext(
        dataframe=dataframe,
        audit_report=None,
        report_html_path=None,
        export_dir=tmp_path,
        warnings=[],
    )

    artifact = JSONExporter().export(context)

    assert artifact.exists is True
    assert (tmp_path / "json" / "export_dataset.json").read_text(encoding="utf-8") == (
        "[\n"
        "  {\n"
        '    "b": 1.0,\n'
        '    "a": "x"\n'
        "  },\n"
        "  {\n"
        '    "b": null,\n'
        '    "a": "y"\n'
        "  }\n"
        "]\n"
    )


def test_json_exporter_handles_missing_dataset_without_exception(tmp_path):
    context = ExportContext(
        dataframe=None,
        audit_report=None,
        report_html_path=None,
        export_dir=tmp_path,
        warnings=[],
    )

    artifact = JSONExporter().export(context)

    assert artifact.exists is False
    assert artifact.warning == "Dataset export unavailable"
    assert (tmp_path / "json").exists()
