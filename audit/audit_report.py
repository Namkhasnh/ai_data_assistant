from __future__ import annotations

import json
from pathlib import Path

from audit.change_log import DEFAULT_ARTIFACT_SPECS, ArtifactSpec, ChangeLogStore, inspect_artifacts
from audit.rule_history import RuleHistoryStore
from models.audit import AuditReport, ChangeLog, RuleHistory


class AuditReportStore:
    """Passive JSON store for complete audit reports."""

    def __init__(
        self,
        audit_path: str | Path = "storage/audit/audit_report.json",
        change_log_path: str | Path = "storage/audit/change_log.json",
        rule_history_path: str | Path = "storage/audit/rule_history.json",
    ) -> None:
        self.audit_path = Path(audit_path)
        self.change_log_path = Path(change_log_path)
        self.rule_history_path = Path(rule_history_path)

    def build(
        self,
        artifact_specs: tuple[ArtifactSpec, ...] = DEFAULT_ARTIFACT_SPECS,
    ) -> AuditReport:
        """Assemble an audit report from existing artifacts without mutating them."""

        artifacts, warnings = inspect_artifacts(artifact_specs)
        change_log = self._load_change_log(warnings)
        rule_history = self._load_rule_history(warnings)

        return AuditReport(
            warnings=warnings,
            artifacts=sorted(artifacts, key=lambda artifact: artifact.artifact),
            change_log=ChangeLog(
                records=sorted(
                    change_log.records,
                    key=lambda record: (
                        record.timestamp,
                        record.module,
                        record.artifact,
                        record.status,
                        record.message,
                    ),
                )
            ),
            rule_history=RuleHistory(
                records=sorted(rule_history.records, key=lambda record: record.rule_id)
            ),
        )

    def write(self, report: AuditReport) -> Path:
        """Persist an audit report without changing source audit files."""

        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return self.audit_path

    def persist(
        self,
        artifact_specs: tuple[ArtifactSpec, ...] = DEFAULT_ARTIFACT_SPECS,
    ) -> tuple[AuditReport, Path]:
        """Build and persist audit_report.json."""

        report = self.build(artifact_specs=artifact_specs)
        output_path = self.write(report)
        return report, output_path

    def _load_change_log(self, warnings: list[str]) -> ChangeLog:
        if not self.change_log_path.exists():
            warnings.append(f"Missing change log: {self.change_log_path}")
            return ChangeLog()
        return ChangeLogStore(audit_path=self.change_log_path).load()

    def _load_rule_history(self, warnings: list[str]) -> RuleHistory:
        if not self.rule_history_path.exists():
            warnings.append(f"Missing rule history: {self.rule_history_path}")
            return RuleHistory()
        return RuleHistoryStore(audit_path=self.rule_history_path).load()
