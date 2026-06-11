from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ArtifactFile:
    """UI-safe artifact metadata."""

    name: str
    path: Path
    size_bytes: int


class ArtifactController:
    """Read artifact files and build UI download lists without mutating them."""

    def __init__(
        self,
        artifact_dir: str | Path = "storage/artifacts",
        audit_dir: str | Path = "storage/audit",
        report_dir: str | Path = "storage/reports",
        export_dir: str | Path = "storage/exports",
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.audit_dir = Path(audit_dir)
        self.report_dir = Path(report_dir)
        self.export_dir = Path(export_dir)

    def artifact_exists(self, path: str | Path) -> bool:
        return Path(path).exists()

    def list_exports(self) -> list[ArtifactFile]:
        return self._list_files(self.export_dir, patterns=("*.csv", "*.xlsx", "*.json", "*.pdf"))

    def list_reports(self) -> list[ArtifactFile]:
        return self._list_files(self.report_dir, patterns=("*.html", "*.json"))

    def list_audit_files(self) -> list[ArtifactFile]:
        return self._list_files(self.audit_dir, patterns=("*.json",))

    def read_json(self, path: str | Path) -> tuple[dict[str, Any] | None, str | None]:
        file_path = Path(path)
        if not file_path.exists():
            return None, f"Missing artifact: {file_path}"
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return None, f"Invalid JSON artifact {file_path}: {exc}"
        if not isinstance(payload, dict):
            return None, f"JSON artifact must contain an object: {file_path}"
        return payload, None

    def read_dataframe(self, path: str | Path) -> tuple[pd.DataFrame | None, str | None]:
        file_path = Path(path)
        if not file_path.exists():
            return None, f"Missing artifact: {file_path}"
        try:
            return pd.read_csv(file_path), None
        except Exception as exc:  # noqa: BLE001 - UI must degrade gracefully.
            return None, f"Unable to read dataframe artifact {file_path}: {exc}"

    def read_text(self, path: str | Path) -> tuple[str | None, str | None]:
        file_path = Path(path)
        if not file_path.exists():
            return None, f"Missing artifact: {file_path}"
        return file_path.read_text(encoding="utf-8"), None

    def _list_files(self, directory: Path, patterns: tuple[str, ...]) -> list[ArtifactFile]:
        if not directory.exists():
            return []
        files: list[Path] = []
        for pattern in patterns:
            files.extend(directory.rglob(pattern))
        return [
            ArtifactFile(name=path.name, path=path, size_bytes=path.stat().st_size)
            for path in sorted(set(files), key=lambda item: str(item))
        ]
