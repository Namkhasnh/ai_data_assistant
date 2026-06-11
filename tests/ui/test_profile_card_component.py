from __future__ import annotations

from models.column_profile import ColumnProfile
from models.dataset import DatasetMetadata

from app.components.profile_card import build_profile_metrics
from app.components.status_badge import badge_html
from app.components.warning_panel import normalize_warnings


def test_profile_card_metrics() -> None:
    metadata = DatasetMetadata(
        source_file="sample.csv",
        file_format="csv",
        row_count=3,
        column_count=2,
        duplicate_count=1,
        columns=[
            ColumnProfile(
                name="a",
                data_type="int64",
                null_count=1,
                null_percentage=33.3,
                unique_value_count=2,
            ),
            ColumnProfile(
                name="b",
                data_type="object",
                null_count=2,
                null_percentage=66.6,
                unique_value_count=1,
            ),
        ],
    )

    metrics = build_profile_metrics(metadata)

    assert metrics.row_count == 3
    assert metrics.column_count == 2
    assert metrics.duplicate_count == 1
    assert metrics.missing_values == 3


def test_warning_panel_and_status_badge_helpers() -> None:
    assert normalize_warnings(["b", "a", "b"]) == ["b", "a"]
    assert "Success" in badge_html("SUCCESS")
    assert "Failed" in badge_html("FAILED")
