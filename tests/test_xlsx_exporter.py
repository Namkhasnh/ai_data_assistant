from __future__ import annotations

import pandas as pd

from core.exporting.base_exporter import ExportContext
from core.exporting.xlsx_exporter import XLSXExporter


def test_xlsx_exporter_uses_deterministic_sheet_order_and_preserves_columns(tmp_path):
    dataframe = pd.DataFrame({"b": [1], "a": ["x"]})
    context = ExportContext(
        dataframe=dataframe,
        audit_report={
            "artifacts": [
                {"artifact": "z.json", "module": "z", "exists": True, "size_bytes": 2},
                {"artifact": "a.json", "module": "a", "exists": True, "size_bytes": 1},
            ]
        },
        report_html_path=None,
        export_dir=tmp_path,
        warnings=["upstream warning"],
    )

    artifact = XLSXExporter().export(context)

    assert artifact.exists is True
    workbook = pd.ExcelFile(tmp_path / "xlsx" / "export_dataset.xlsx")
    assert workbook.sheet_names == ["Dataset", "Audit Summary", "Warnings"]
    exported = pd.read_excel(workbook, sheet_name="Dataset")
    assert list(exported.columns) == ["b", "a"]
    audit_summary = pd.read_excel(workbook, sheet_name="Audit Summary")
    assert audit_summary["artifact"].tolist() == ["a.json", "z.json"]


def test_xlsx_exporter_handles_missing_dataset_without_exception(tmp_path):
    context = ExportContext(
        dataframe=None,
        audit_report=None,
        report_html_path=None,
        export_dir=tmp_path,
        warnings=[],
    )

    artifact = XLSXExporter().export(context)

    assert artifact.exists is False
    assert artifact.warning == "Dataset export unavailable"
    assert (tmp_path / "xlsx").exists()
