from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from models.audit import ArtifactRecord, AuditRecord, ChangeLog, utc_now


@dataclass(frozen=True)
class ArtifactSpec:
    """Artifact path and owning module used for passive audit inspection."""

    module: str
    artifact: str
    path: Path


DEFAULT_ARTIFACT_SPECS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("profiling", "metadata.json", Path("storage/artifacts/metadata.json")),
    ArtifactSpec("semantic", "semantic_columns.json", Path("storage/artifacts/semantic_columns.json")),
    ArtifactSpec("rule_generation", "rules.json", Path("data/rules/rules.json")),
    ArtifactSpec("rule_execution", "rule_execution_report.json", Path("storage/artifacts/rule_execution_report.json")),
    ArtifactSpec("standardization", "standardization_report.json", Path("storage/artifacts/standardization_report.json")),
    ArtifactSpec("standardization", "standardized_dataset.csv", Path("storage/artifacts/standardized_dataset.csv")),
    ArtifactSpec("enrichment", "enrichment_report.json", Path("storage/artifacts/enrichment_report.json")),
    ArtifactSpec("enrichment", "enriched_dataset.csv", Path("storage/artifacts/enriched_dataset.csv")),
    ArtifactSpec("ai_suggestions", "ai_suggestions.json", Path("storage/artifacts/ai_suggestions.json")),
)


class ChangeLogStore:
    """Append-only JSON store for audit change records."""

    def __init__(self, audit_path: str | Path = "storage/audit/change_log.json") -> None:
        self.audit_path = Path(audit_path)

    def load(self) -> ChangeLog:
        """Load an existing change log or return an empty log if it is missing."""

        if not self.audit_path.exists():
            return ChangeLog()
        payload = self._read_json(self.audit_path)
        return ChangeLog.model_validate(payload)

    def write(self, change_log: ChangeLog) -> Path:
        """Write the change log sorted by timestamp."""

        sorted_records = sorted(
            change_log.records,
            key=lambda record: (
                record.timestamp,
                record.module,
                record.artifact,
                record.status,
                record.message,
            ),
        )
        output = ChangeLog(records=sorted_records)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.write_text(
            json.dumps(output.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return self.audit_path

    def append(self, record: AuditRecord) -> ChangeLog:
        """Append one audit record and persist the full log."""

        change_log = self.load()
        change_log.records.append(record)
        self.write(change_log)
        return self.load()

    def record(
        self,
        module: str,
        artifact: str,
        status: str,
        message: str,
        timestamp: datetime | None = None,
    ) -> ChangeLog:
        """Create and append one audit record."""

        record = AuditRecord(
            timestamp=timestamp or utc_now(),
            module=module,
            artifact=artifact,
            status=status,
            message=message,
        )
        return self.append(record)

    def build_from_artifacts(
        self,
        artifact_specs: tuple[ArtifactSpec, ...] = DEFAULT_ARTIFACT_SPECS,
        timestamp: datetime | None = None,
    ) -> tuple[ChangeLog, list[ArtifactRecord], list[str]]:
        """Build a change log from artifact existence and metadata."""

        observed_at = timestamp or utc_now()
        artifacts, warnings = inspect_artifacts(artifact_specs)
        records = [
            AuditRecord(
                timestamp=observed_at,
                module=artifact.module,
                artifact=artifact.artifact,
                status="success" if artifact.exists else "missing",
                message=(
                    "Artifact present"
                    if artifact.exists
                    else f"Artifact missing: {artifact.path}"
                ),
            )
            for artifact in artifacts
        ]
        change_log = self.load()
        change_log.records.extend(records)
        self.write(change_log)
        return self.load(), artifacts, warnings

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Audit file is invalid JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Audit file must contain a JSON object: {path}")
        return payload


def inspect_artifacts(
    artifact_specs: tuple[ArtifactSpec, ...] = DEFAULT_ARTIFACT_SPECS,
) -> tuple[list[ArtifactRecord], list[str]]:
    """Return deterministic artifact records plus warnings for missing files."""

    records: list[ArtifactRecord] = []
    warnings: list[str] = []
    for spec in sorted(artifact_specs, key=lambda item: item.artifact):
        if not spec.path.exists():
            warnings.append(f"Missing artifact: {spec.path}")
            records.append(
                ArtifactRecord(
                    artifact=spec.artifact,
                    path=str(spec.path),
                    module=spec.module,
                    exists=False,
                )
            )
            continue

        stat = spec.path.stat()
        records.append(
            ArtifactRecord(
                artifact=spec.artifact,
                path=str(spec.path),
                module=spec.module,
                exists=True,
                content_hash=sha256_file(spec.path),
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            )
        )
    return records, warnings


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 hash for a file without modifying it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
