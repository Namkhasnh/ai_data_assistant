from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.ai.base_provider import BaseProvider
from core.ai.ollama_provider import OllamaProvider
from core.ai.prompt_builder import PromptBuilder
from core.ai.response_parser import AIResponseParseError, ResponseParser
from core.ai.validation_layer import ValidationLayer
from models.ai_suggestion import AISuggestionReport
from models.dataset import DatasetMetadata
from models.semantic_tag import SemanticDetectionReport
from storage.artifact_store import ArtifactStore


logger = logging.getLogger(__name__)


class AIRuleGenerationService:
    """Generate LLM suggestions without executing or persisting rules."""

    def __init__(
        self,
        provider: BaseProvider | None = None,
        prompt_builder: PromptBuilder | None = None,
        response_parser: ResponseParser | None = None,
        validation_layer: ValidationLayer | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.provider = provider or OllamaProvider()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.response_parser = response_parser or ResponseParser()
        self.validation_layer = validation_layer or ValidationLayer()
        self.artifact_store = artifact_store or ArtifactStore()

    def generate_suggestions(
        self,
        metadata_path: str | Path = "storage/artifacts/metadata.json",
        semantic_report_path: str | Path = "storage/artifacts/semantic_columns.json",
        output_filename: str = "ai_suggestions.json",
    ) -> AISuggestionReport:
        """Generate validated AI suggestions and write ai_suggestions.json."""

        metadata = self.load_metadata(metadata_path)
        semantic_report = self.load_semantic_report(semantic_report_path)

        if not self._provider_available():
            report = self.validation_layer.empty_report(
                provider=self.provider.name,
                model=self.provider.model,
                warnings=[f"AI provider unavailable: {self.provider.name}"],
            )
            self.artifact_store.write_json(output_filename, report)
            return report

        prompt = self.prompt_builder.build_rule_generation_prompt(metadata, semantic_report)
        try:
            raw_response = self.provider.generate(prompt)
        except Exception as exc:
            logger.warning("AI provider generation failed: %s", exc)
            report = self.validation_layer.empty_report(
                provider=self.provider.name,
                model=self.provider.model,
                warnings=[f"AI provider generation failed: {exc}"],
            )
            self.artifact_store.write_json(output_filename, report)
            return report

        try:
            suggestions = self.response_parser.parse(raw_response)
        except AIResponseParseError as exc:
            logger.warning("AI suggestion parsing failed: %s", exc)
            report = self.validation_layer.empty_report(
                provider=self.provider.name,
                model=self.provider.model,
                warnings=[f"AI suggestion parsing failed: {exc}"],
            )
            self.artifact_store.write_json(output_filename, report)
            return report

        report = self.validation_layer.validate(
            suggestions=suggestions,
            provider=self.provider.name,
            model=self.provider.model,
        )
        self.artifact_store.write_json(output_filename, report)
        return report

    @staticmethod
    def load_metadata(metadata_path: str | Path) -> DatasetMetadata:
        payload = AIRuleGenerationService._read_json(Path(metadata_path))
        return DatasetMetadata.model_validate(payload)

    @staticmethod
    def load_semantic_report(semantic_report_path: str | Path) -> SemanticDetectionReport:
        payload = AIRuleGenerationService._read_json(Path(semantic_report_path))
        return SemanticDetectionReport.model_validate(payload)

    def _provider_available(self) -> bool:
        try:
            return self.provider.health_check()
        except Exception as exc:
            logger.warning("AI provider health check failed: %s", exc)
            return False

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"AI input artifact not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"AI input artifact is invalid JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"AI input artifact must be a JSON object: {path}")
        return payload
