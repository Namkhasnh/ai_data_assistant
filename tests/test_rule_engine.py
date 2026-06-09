from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.rules.rule_engine import RuleEngine
from models.rule import Rule, RuleSet
from services.rule_service import RuleService
from storage.artifact_store import ArtifactStore


def test_rule_engine_executes_rules_by_priority_sequentially_without_mutation() -> None:
    dataframe = pd.DataFrame({"location": [" hn ", "Ha Noi", "Other"]})
    original = dataframe.copy(deep=True)
    rule_set = RuleSet(
        rules=[
            Rule(
                id="map_location",
                type="mapping",
                column="location",
                priority=20,
                parameters={
                    "mapping": {
                        "hn": "Hanoi",
                        "Ha Noi": "Hanoi",
                    },
                    "case_sensitive": False,
                },
            ),
            Rule(
                id="trim_location",
                type="transformation",
                column="location",
                priority=10,
                parameters={"operation": "trim"},
            ),
            Rule(
                id="disabled_validation",
                type="validation",
                column="location",
                enabled=False,
                priority=30,
                parameters={"pattern": r".+"},
            ),
        ]
    )

    cleaned, report = RuleEngine().execute(dataframe, rule_set)

    assert cleaned["location"].tolist() == ["Hanoi", "Hanoi", "Other"]
    pd.testing.assert_frame_equal(dataframe, original)
    assert [result.rule_id for result in report.results] == [
        "trim_location",
        "map_location",
        "disabled_validation",
    ]
    assert [result.status for result in report.results] == [
        "applied",
        "applied",
        "skipped",
    ]
    assert report.results[0].affected_rows == 1
    assert report.results[1].affected_rows == 2


def test_rule_engine_reports_failed_rule_and_keeps_dataframe_unchanged_for_that_rule() -> None:
    dataframe = pd.DataFrame({"value": ["a"]})
    rule_set = RuleSet(
        rules=[
            Rule(
                id="bad_mapping",
                type="mapping",
                column="missing",
                parameters={"mapping": {"a": "b"}},
            )
        ]
    )

    cleaned, report = RuleEngine().execute(dataframe, rule_set)

    pd.testing.assert_frame_equal(cleaned, dataframe)
    assert report.results[0].status == "failed"
    assert report.results[0].affected_rows == 0
    assert "Column not found" in report.results[0].message


def test_rule_engine_counts_rows_affected_by_new_regex_columns() -> None:
    dataframe = pd.DataFrame({"range": ["1-2", "none", "3-4"]})
    rule_set = RuleSet(
        rules=[
            Rule(
                id="extract_range",
                type="regex",
                column="range",
                parameters={
                    "pattern": r"(\d+)-(\d+)",
                    "output_columns": ["min_value", "max_value"],
                    "output_types": {
                        "min_value": "int",
                        "max_value": "int",
                    },
                },
            )
        ]
    )

    cleaned, report = RuleEngine().execute(dataframe, rule_set)

    assert cleaned["min_value"].tolist()[:1] == [1]
    assert report.results[0].affected_rows == 2


def test_rule_service_loads_rules_executes_engine_and_persists_artifacts(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "id": "trim_title",
                        "type": "transformation",
                        "column": "title",
                        "parameters": {"operation": "trim"},
                        "priority": 10,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    dataframe = pd.DataFrame({"title": ["  Data Engineer  "]})
    artifact_store = ArtifactStore(artifact_dir=tmp_path / "artifacts")

    cleaned, report = RuleService(artifact_store=artifact_store).execute_rules(
        df=dataframe,
        rules_path=rules_path,
    )

    assert cleaned["title"].tolist() == ["Data Engineer"]
    assert report.results[0].status == "applied"
    assert (tmp_path / "artifacts" / "cleaned_dataset.csv").exists()
    assert (tmp_path / "artifacts" / "rule_execution_report.json").exists()
