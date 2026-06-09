from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.rule_generation.rule_generation_engine import RuleGenerationEngine
from models.dataset import DatasetMetadata
from models.rule import RuleSet
from models.rule_generation import RuleGenerationReport
from models.semantic_tag import SemanticDetectionReport
from storage.artifact_store import ArtifactStore


class RuleGenerationService:
    """Service layer for deterministic rule generation workflows."""

    def __init__(
        self,
        engine: RuleGenerationEngine | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.engine = engine or RuleGenerationEngine()
        self.artifact_store = artifact_store or ArtifactStore()

    def generate_from_artifacts(
        self,
        metadata_path: str | Path = "storage/artifacts/metadata.json",
        semantic_report_path: str | Path = "storage/artifacts/semantic_columns.json",
        rules_output_path: str | Path = "data/rules/rules.json",
        report_filename: str = "rule_generation_report.json",
    ) -> tuple[RuleSet, RuleGenerationReport]:
        metadata = self.load_metadata(metadata_path)
        semantic_report = self.load_semantic_report(semantic_report_path)
        rule_set, report = self.engine.generate(metadata, semantic_report)
        self.write_rules(rule_set, rules_output_path)
        self.artifact_store.write_json(report_filename, report)
        return rule_set, report

    @staticmethod
    def load_metadata(metadata_path: str | Path) -> DatasetMetadata:
        payload = RuleGenerationService._read_json(Path(metadata_path))
        return DatasetMetadata.model_validate(payload)

    @staticmethod
    def load_semantic_report(semantic_report_path: str | Path) -> SemanticDetectionReport:
        payload = RuleGenerationService._read_json(Path(semantic_report_path))
        return SemanticDetectionReport.model_validate(payload)

    @staticmethod
    def write_rules(rule_set: RuleSet, rules_output_path: str | Path) -> Path:
        output_path = Path(rules_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(rule_set.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output_path

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Input artifact not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Input artifact is invalid JSON: {path}") from exc
