from __future__ import annotations

import pandas as pd

from core.exporting.base_exporter import ExportContext
from core.exporting.csv_exporter import CSVExporter


def test_csv_exporter_preserves_column_order_and_does_not_mutate_dataframe(tmp_path):
    dataframe = pd.DataFrame({"b": [1, 2], "a": ["x", "y"]})
    original = dataframe.copy(deep=True)
    context = ExportContext(
        dataframe=dataframe,
        audit_report=None,
        report_html_path=None,
        export_dir=tmp_path,
        warnings=[],
    )

    artifact = CSVExporter().export(context)

    assert artifact.exists is True
    assert artifact.warning is None
    assert (tmp_path / "csv" / "export_dataset.csv").read_text(encoding="utf-8").splitlines()[0] == "b,a"
    pd.testing.assert_frame_equal(dataframe, original)


def test_csv_exporter_handles_missing_dataset_without_exception(tmp_path):
    context = ExportContext(
        dataframe=None,
        audit_report=None,
        report_html_path=None,
        export_dir=tmp_path,
        warnings=[],
    )

    artifact = CSVExporter().export(context)

    assert artifact.exists is False
    assert artifact.warning == "Dataset export unavailable"
    assert (tmp_path / "csv").exists()
