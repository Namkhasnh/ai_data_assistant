from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.standardization.mapping_standardizer import MappingStandardizer
from models.semantic_tag import SemanticDetectionReport, SemanticTag


def test_mapping_standardizer_uses_knowledge_file_and_preserves_unknowns(tmp_path: Path) -> None:
    knowledge_path = tmp_path / "cities.json"
    knowledge_path.write_text(
        json.dumps(
            {
                "Hà Nội": {
                    "aliases": ["HN", "Ha Noi"],
                }
            }
        ),
        encoding="utf-8",
    )
    dataframe = pd.DataFrame({"location": ["HN", "Ha Noi", "Unknown"]})
    original = dataframe.copy(deep=True)
    semantic_report = SemanticDetectionReport(
        source_file="sample.csv",
        column_count=1,
        columns=[
            SemanticTag(
                column_name="location",
                semantic_type="LOCATION",
                confidence=0.9,
                detector_name="test",
            )
        ],
    )
    config = {
        "LOCATION": {
            "standardizer": "mapping",
            "knowledge_file": str(knowledge_path),
        }
    }

    standardized = MappingStandardizer().standardize(dataframe, semantic_report, config)

    assert standardized["location"].tolist() == ["Hà Nội", "Hà Nội", "Unknown"]
    pd.testing.assert_frame_equal(dataframe, original)
