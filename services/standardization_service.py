from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from core.standardization.standardization_engine import StandardizationEngine
from models.semantic_tag import SemanticDetectionReport
from models.standardization import StandardizationReport
from storage.artifact_store import ArtifactStore


class StandardizationService:
    """Service layer for deterministic standardization workflows."""

    def __init__(
        self,
        engine: StandardizationEngine | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.engine = engine or StandardizationEngine()
        self.artifact_store = artifact_store or ArtifactStore()

    def standardize_dataframe(
        self,
        dataframe: pd.DataFrame,
        semantic_report_path: str | Path = "storage/artifacts/semantic_columns.json",
        config_path: str | Path = "knowledge/domains/jobs/standardization_rules.json",
        standardized_filename: str = "standardized_dataset.csv",
        report_filename: str = "standardization_report.json",
    ) -> tuple[pd.DataFrame, StandardizationReport]:
        semantic_report = self.load_semantic_report(semantic_report_path)
        config = self.load_config(config_path)
        standardized_dataframe, report = self.engine.standardize(
            dataframe=dataframe,
            semantic_report=semantic_report,
            standardization_config=config,
        )
        self.artifact_store.write_dataframe_csv(standardized_filename, standardized_dataframe)
        self.artifact_store.write_json(report_filename, report)
        return standardized_dataframe, report

    def standardize_csv(
        self,
        input_csv_path: str | Path = "storage/artifacts/cleaned_dataset.csv",
        semantic_report_path: str | Path = "storage/artifacts/semantic_columns.json",
        config_path: str | Path = "knowledge/domains/jobs/standardization_rules.json",
        standardized_filename: str = "standardized_dataset.csv",
        report_filename: str = "standardization_report.json",
    ) -> tuple[pd.DataFrame, StandardizationReport]:
        dataframe = pd.read_csv(input_csv_path)
        return self.standardize_dataframe(
            dataframe=dataframe,
            semantic_report_path=semantic_report_path,
            config_path=config_path,
            standardized_filename=standardized_filename,
            report_filename=report_filename,
        )

    @staticmethod
    def load_semantic_report(semantic_report_path: str | Path) -> SemanticDetectionReport:
        payload = StandardizationService._read_json(Path(semantic_report_path))
        return SemanticDetectionReport.model_validate(payload)

    @staticmethod
    def load_config(config_path: str | Path) -> dict[str, Any]:
        payload = StandardizationService._read_json(Path(config_path))
        if not isinstance(payload, dict):
            raise ValueError(f"Standardization config must be a JSON object: {config_path}")
        return payload

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Standardization input not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Standardization input is invalid JSON: {path}") from exc
