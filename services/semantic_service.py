from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.semantic.base_detector import SemanticDetectionInput
from core.semantic.detector_registry import SemanticDetectorRegistry
from models.column_profile import ColumnProfile
from models.dataset import DatasetMetadata
from models.semantic_tag import SemanticDetectionReport
from storage.artifact_store import ArtifactStore


class SemanticService:
    """Thin service layer for deterministic semantic detection."""

    def __init__(
        self,
        registry: SemanticDetectorRegistry | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.registry = registry or SemanticDetectorRegistry.from_knowledge_dir()
        self.artifact_store = artifact_store or ArtifactStore()

    def detect_columns(
        self,
        metadata: DatasetMetadata,
        output_path: str | Path | None = None,
        write_output: bool = True,
    ) -> SemanticDetectionReport:
        detection_inputs = [
            self._build_detection_input(column)
            for column in metadata.columns
        ]
        tags = self.registry.detect_many(detection_inputs)
        report = SemanticDetectionReport(
            source_file=metadata.source_file,
            column_count=len(tags),
            columns=tags,
        )

        if write_output:
            if output_path is not None:
                self.artifact_store.write_json_to_path(output_path, report)
            else:
                self.artifact_store.write_json("semantic_columns.json", report)

        return report

    def detect_from_metadata_file(
        self,
        metadata_path: str | Path,
        output_path: str | Path | None = None,
        write_output: bool = True,
    ) -> SemanticDetectionReport:
        metadata = self._load_metadata(Path(metadata_path))
        return self.detect_columns(
            metadata=metadata,
            output_path=output_path,
            write_output=write_output,
        )

    @staticmethod
    def _build_detection_input(column: ColumnProfile) -> SemanticDetectionInput:
        return SemanticDetectionInput(
            column_name=column.name,
            data_type=column.data_type,
            top_values=[
                SemanticService._stringify_value(top_value.value)
                for top_value in column.top_values
            ],
            sample_values=[
                SemanticService._stringify_value(sample_value)
                for sample_value in column.sample_values
            ],
        )

    @staticmethod
    def _load_metadata(metadata_path: Path) -> DatasetMetadata:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Metadata file is invalid JSON: {metadata_path}") from exc

        return DatasetMetadata.model_validate(payload)

    @staticmethod
    def _stringify_value(value: Any) -> str:
        if value is None:
            return ""
        return str(value)
