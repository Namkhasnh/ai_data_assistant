from __future__ import annotations

import pandas as pd

from core.reporting.plotly_chart_builder import PlotlyChartBuilder


def test_chart_builder_generates_stable_sorted_chart_assets(tmp_path):
    dataframe = pd.DataFrame(
        {
            "b_numeric": [3, 4, 5],
            "a_numeric": [1, 2, 3],
            "z_category": ["beta", "alpha", "beta"],
            "a_category": ["x", "y", "x"],
        }
    )
    semantic_payload = {
        "columns": [
            {"column_name": "a_category", "semantic_type": "CATEGORY"},
            {"column_name": "a_numeric", "semantic_type": "NUMBER"},
        ]
    }
    rules_payload = {
        "rules": [
            {"type": "regex"},
            {"type": "mapping"},
            {"type": "mapping"},
        ]
    }
    audit_payload = {
        "artifacts": [
            {"artifact": "z.json", "exists": True, "size_bytes": 20},
            {"artifact": "a.json", "exists": True, "size_bytes": 10},
        ]
    }
    warnings: list[str] = []

    charts = PlotlyChartBuilder(max_numeric_charts=1, max_categorical_charts=1).build_charts(
        dataframe=dataframe,
        semantic_payload=semantic_payload,
        rules_payload=rules_payload,
        audit_payload=audit_payload,
        output_dir=tmp_path,
        warnings=warnings,
    )

    chart_ids = [chart.chart_id for chart in charts]
    assert chart_ids == sorted(chart_ids)
    assert "numeric-a-numeric" in chart_ids
    assert "numeric-b-numeric" not in chart_ids
    assert "categorical-a-category" in chart_ids
    assert "categorical-z-category" not in chart_ids
    assert (tmp_path / "semantic_distribution.html").exists()
    assert (tmp_path / "rule_distribution.html").exists()
    assert (tmp_path / "artifact_sizes.html").exists()
    assert all((tmp_path / chart.path.removeprefix("assets/")).exists() for chart in charts)
    assert warnings == []
