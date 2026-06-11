from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

import pandas as pd

from app.controllers.session_controller import SessionController


class WorkspaceController:
    """Manage app run paths, uploaded files, and workspace-local artifacts."""

    def __init__(
        self,
        session: SessionController,
        workspace_root: str | Path = "storage/app_runs",
    ) -> None:
        self.session = session
        self.workspace_root = Path(workspace_root)

    def ensure_run_id(self) -> str:
        run_id = self.session.get("run_id")
        if run_id:
            return str(run_id)
        generated = self.create_run_id()
        self.session.set("run_id", generated)
        return generated

    def create_run_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"run_{timestamp}_{uuid4().hex[:8]}"

    @property
    def run_dir(self) -> Path:
        return self.workspace_root / self.ensure_run_id()

    @property
    def upload_dir(self) -> Path:
        return self.run_dir / "uploads"

    @property
    def artifact_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def audit_dir(self) -> Path:
        return self.run_dir / "audit"

    @property
    def report_dir(self) -> Path:
        return self.run_dir / "reports"

    @property
    def export_dir(self) -> Path:
        return self.run_dir / "exports"

    @property
    def rules_path(self) -> Path:
        return self.run_dir / "rules" / "rules.json"

    def save_upload(self, uploaded_file: BinaryIO, filename: str) -> Path:
        safe_name = Path(filename).name
        output_path = self.upload_dir / safe_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        uploaded_file.seek(0)
        output_path.write_bytes(uploaded_file.read())
        return output_path

    def load_dataframe(self, file_path: str | Path) -> pd.DataFrame:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError(f"Unsupported upload format: {suffix}")
