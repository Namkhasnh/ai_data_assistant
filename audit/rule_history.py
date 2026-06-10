from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from models.audit import RuleHistory, RuleHistoryRecord, utc_now
from models.rule import ExecutionReport, RuleSet


class RuleHistoryStore:
    """Passive JSON store for aggregated rule history."""

    def __init__(self, audit_path: str | Path = "storage/audit/rule_history.json") -> None:
        self.audit_path = Path(audit_path)

    def load(self) -> RuleHistory:
        """Load existing rule history or return an empty history."""

        if not self.audit_path.exists():
            return RuleHistory()
        payload = self._read_json(self.audit_path)
        return RuleHistory.model_validate(payload)

    def write(self, rule_history: RuleHistory) -> Path:
        """Write rule history sorted by rule ID."""

        sorted_records = sorted(rule_history.records, key=lambda record: record.rule_id)
        output = RuleHistory(records=sorted_records)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.write_text(
            json.dumps(output.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return self.audit_path

    def build_from_files(
        self,
        rules_path: str | Path = "data/rules/rules.json",
        execution_report_path: str | Path = "storage/artifacts/rule_execution_report.json",
        observed_at: datetime | None = None,
        merge_existing: bool = True,
    ) -> tuple[RuleHistory, list[str]]:
        """Build rule history from current rules and rule execution artifacts."""

        timestamp = observed_at or utc_now()
        warnings: list[str] = []
        existing_by_id = {
            record.rule_id: record
            for record in self.load().records
        } if merge_existing else {}

        rule_set, rule_warnings = self._load_rule_set(Path(rules_path))
        execution_report, report_warnings = self._load_execution_report(Path(execution_report_path))
        warnings.extend(rule_warnings)
        warnings.extend(report_warnings)

        execution_counts: dict[str, int] = {}
        execution_types: dict[str, str] = {}
        if execution_report is not None:
            for result in execution_report.results:
                execution_counts[result.rule_id] = execution_counts.get(result.rule_id, 0) + 1
                execution_types[result.rule_id] = result.rule_type

        records_by_id: dict[str, RuleHistoryRecord] = {}
        if rule_set is not None:
            for rule in rule_set.rules:
                existing = existing_by_id.get(rule.id)
                previous_count = existing.execution_count if existing else 0
                current_count = execution_counts.get(rule.id, 0)
                records_by_id[rule.id] = RuleHistoryRecord(
                    rule_id=rule.id,
                    rule_type=rule.type,
                    created_by=rule.created_by,
                    enabled=rule.enabled,
                    execution_count=previous_count + current_count,
                    first_seen=existing.first_seen if existing else timestamp,
                    last_seen=timestamp,
                )

        for rule_id, count in execution_counts.items():
            if rule_id in records_by_id:
                continue
            warnings.append(f"Execution report contains rule not present in rules file: {rule_id}")
            existing = existing_by_id.get(rule_id)
            previous_count = existing.execution_count if existing else 0
            records_by_id[rule_id] = RuleHistoryRecord(
                rule_id=rule_id,
                rule_type=execution_types.get(rule_id, "unknown"),
                created_by=existing.created_by if existing else "unknown",
                enabled=existing.enabled if existing else False,
                execution_count=previous_count + count,
                first_seen=existing.first_seen if existing else timestamp,
                last_seen=timestamp,
            )

        for rule_id, existing in existing_by_id.items():
            records_by_id.setdefault(rule_id, existing)

        rule_history = RuleHistory(records=sorted(records_by_id.values(), key=lambda record: record.rule_id))
        self.write(rule_history)
        return self.load(), warnings

    @staticmethod
    def _load_rule_set(path: Path) -> tuple[RuleSet | None, list[str]]:
        if not path.exists():
            return None, [f"Missing rules file: {path}"]
        try:
            payload = RuleHistoryStore._read_json(path)
            return RuleSet.model_validate(payload), []
        except (ValueError, TypeError) as exc:
            return None, [f"Unable to read rules file {path}: {exc}"]

    @staticmethod
    def _load_execution_report(path: Path) -> tuple[ExecutionReport | None, list[str]]:
        if not path.exists():
            return None, [f"Missing rule execution report: {path}"]
        try:
            payload = RuleHistoryStore._read_json(path)
            return ExecutionReport.model_validate(payload), []
        except (ValueError, TypeError) as exc:
            return None, [f"Unable to read rule execution report {path}: {exc}"]

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Audit source is invalid JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Audit source must contain a JSON object: {path}")
        return payload
