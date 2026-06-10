from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from core.enrichment.enrichment_engine import EnrichmentEngine
from models.enrichment import EnrichmentConfig, EnrichmentReport
from models.semantic_tag import SemanticDetectionReport
from storage.artifact_store import ArtifactStore


class EnrichmentService:
    """Service layer for deterministic knowledge-based enrichment workflows."""

    def __init__(
        self,
        engine: EnrichmentEngine | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.engine = engine or EnrichmentEngine()
        self.artifact_store = artifact_store or ArtifactStore()

    def enrich_dataframe(
        self,
        dataframe: pd.DataFrame,
        semantic_report_path: str | Path = "storage/artifacts/semantic_columns.json",
        config_path: str | Path = "knowledge/domains/jobs/enrichment_rules.json",
        enriched_filename: str = "enriched_dataset.csv",
        report_filename: str = "enrichment_report.json",
    ) -> tuple[pd.DataFrame, EnrichmentReport]:
        semantic_report = self.load_semantic_report(semantic_report_path)
        enrichment_config = self.load_config(config_path)
        enriched_dataframe, report = self.engine.enrich(
            dataframe=dataframe,
            semantic_report=semantic_report,
            enrichment_config=enrichment_config,
        )
        self.artifact_store.write_dataframe_csv(enriched_filename, enriched_dataframe)
        self.artifact_store.write_json(report_filename, report)
        return enriched_dataframe, report

    def enrich_csv(
        self,
        input_csv_path: str | Path = "storage/artifacts/standardized_dataset.csv",
        semantic_report_path: str | Path = "storage/artifacts/semantic_columns.json",
        config_path: str | Path = "knowledge/domains/jobs/enrichment_rules.json",
        enriched_filename: str = "enriched_dataset.csv",
        report_filename: str = "enrichment_report.json",
    ) -> tuple[pd.DataFrame, EnrichmentReport]:
        dataframe = pd.read_csv(input_csv_path)
        return self.enrich_dataframe(
            dataframe=dataframe,
            semantic_report_path=semantic_report_path,
            config_path=config_path,
            enriched_filename=enriched_filename,
            report_filename=report_filename,
        )

    @staticmethod
    def load_semantic_report(semantic_report_path: str | Path) -> SemanticDetectionReport:
        payload = EnrichmentService._read_json(Path(semantic_report_path))
        return SemanticDetectionReport.model_validate(payload)

    @staticmethod
    def load_config(config_path: str | Path) -> EnrichmentConfig:
        payload = EnrichmentService._read_json(Path(config_path))
        return EnrichmentConfig.model_validate(payload)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Enrichment input not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Enrichment input is invalid JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Enrichment input must be a JSON object: {path}")
        return payload
