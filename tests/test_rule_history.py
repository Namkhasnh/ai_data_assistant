from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from audit.change_log import sha256_file
from audit.rule_history import RuleHistoryStore


def test_rule_history_builds_sorted_records_and_execution_counts(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.json"
    report_path = tmp_path / "rule_execution_report.json"
    rules_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "id": "rule_b",
                        "type": "regex",
                        "column": "salary",
                        "parameters": {},
                        "enabled": True,
                        "created_by": "semantic",
                    },
                    {
                        "id": "rule_a",
                        "type": "mapping",
                        "column": "location",
                        "parameters": {},
                        "enabled": False,
                        "created_by": "manual",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "rule_id": "rule_b",
                        "rule_type": "regex",
                        "status": "applied",
                        "affected_rows": 2,
                        "execution_time_ms": 1.0,
                        "message": "Rule applied",
                    },
                    {
                        "rule_id": "rule_b",
                        "rule_type": "regex",
                        "status": "applied",
                        "affected_rows": 3,
                        "execution_time_ms": 1.0,
                        "message": "Rule applied",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    observed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    history, warnings = RuleHistoryStore(
        audit_path=tmp_path / "audit" / "rule_history.json"
    ).build_from_files(
        rules_path=rules_path,
        execution_report_path=report_path,
        observed_at=observed_at,
    )

    assert warnings == []
    assert [record.rule_id for record in history.records] == ["rule_a", "rule_b"]
    assert history.records[0].execution_count == 0
    assert history.records[1].execution_count == 2
    assert history.model_dump(mode="json")["records"][0]["first_seen"] == "2026-01-01T00:00:00Z"


def test_rule_history_missing_files_generate_warnings(tmp_path: Path) -> None:
    history, warnings = RuleHistoryStore(
        audit_path=tmp_path / "audit" / "rule_history.json"
    ).build_from_files(
        rules_path=tmp_path / "missing_rules.json",
        execution_report_path=tmp_path / "missing_report.json",
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert history.records == []
    assert len(warnings) == 2
    assert "Missing rules file" in warnings[0]
    assert "Missing rule execution report" in warnings[1]


def test_rule_history_preserves_existing_records_and_does_not_mutate_sources(
    tmp_path: Path,
) -> None:
    rules_path = tmp_path / "rules.json"
    report_path = tmp_path / "rule_execution_report.json"
    rules_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "id": "rule_a",
                        "type": "mapping",
                        "column": "location",
                        "parameters": {},
                        "enabled": True,
                        "created_by": "semantic",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "rule_id": "rule_a",
                        "rule_type": "mapping",
                        "status": "applied",
                        "affected_rows": 1,
                        "execution_time_ms": 1.0,
                        "message": "Rule applied",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rules_hash = sha256_file(rules_path)
    report_hash = sha256_file(report_path)
    store = RuleHistoryStore(audit_path=tmp_path / "audit" / "rule_history.json")

    first, _ = store.build_from_files(
        rules_path=rules_path,
        execution_report_path=report_path,
        observed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    second, _ = store.build_from_files(
        rules_path=rules_path,
        execution_report_path=report_path,
        observed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    assert first.records[0].execution_count == 1
    assert second.records[0].execution_count == 2
    assert second.records[0].first_seen == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert second.records[0].last_seen == datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert sha256_file(rules_path) == rules_hash
    assert sha256_file(report_path) == report_hash
