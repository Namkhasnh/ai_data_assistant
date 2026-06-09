from __future__ import annotations

from pathlib import Path

from core.ai.base_provider import BaseProvider
from models.column_profile import ColumnProfile, TopValue
from models.dataset import DatasetMetadata
from models.semantic_tag import SemanticDetectionReport, SemanticTag
from services.ai_rule_generation_service import AIRuleGenerationService
from storage.artifact_store import ArtifactStore


class FakeProvider(BaseProvider):
    """Test provider that avoids any Ollama dependency."""

    name = "fake"
    model = "fake-model"

    def __init__(
        self,
        response: str = '{"suggestions": []}',
        available: bool = True,
        fail_generation: bool = False,
    ) -> None:
        self.response = response
        self.available = available
        self.fail_generation = fail_generation
        self.last_prompt: str | None = None

    def health_check(self) -> bool:
        return self.available

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        if self.fail_generation:
            raise RuntimeError("provider failed")
        return self.response


def test_ai_rule_generation_service_writes_validated_suggestions(tmp_path: Path) -> None:
    metadata_path, semantic_path = _write_inputs(tmp_path)
    provider = FakeProvider(
        response="""
        {
          "suggestions": [
            {
              "type": "mapping",
              "column": "title",
              "semantic_type": "JOB_TITLE",
              "source_value": "Machine Learning Engineer",
              "target_value": "AI Engineer",
              "confidence": 0.92,
              "reason": "Similar role naming",
              "created_by": "llm"
            }
          ]
        }
        """
    )
    artifact_store = ArtifactStore(artifact_dir=tmp_path / "artifacts")

    report = AIRuleGenerationService(
        provider=provider,
        artifact_store=artifact_store,
    ).generate_suggestions(
        metadata_path=metadata_path,
        semantic_report_path=semantic_path,
    )

    assert len(report.suggestions) == 1
    assert report.suggestions[0].suggestion_id is not None
    assert provider.last_prompt is not None
    assert "Machine Learning Engineer" in provider.last_prompt
    assert (tmp_path / "artifacts" / "ai_suggestions.json").exists()


def test_ai_rule_generation_service_returns_empty_report_when_provider_unavailable(
    tmp_path: Path,
) -> None:
    metadata_path, semantic_path = _write_inputs(tmp_path)
    provider = FakeProvider(available=False)
    artifact_store = ArtifactStore(artifact_dir=tmp_path / "artifacts")

    report = AIRuleGenerationService(
        provider=provider,
        artifact_store=artifact_store,
    ).generate_suggestions(
        metadata_path=metadata_path,
        semantic_report_path=semantic_path,
    )

    assert report.suggestions == []
    assert "AI provider unavailable: fake" in report.warnings
    assert provider.last_prompt is None
    assert (tmp_path / "artifacts" / "ai_suggestions.json").exists()


def test_ai_rule_generation_service_returns_empty_report_on_provider_failure(
    tmp_path: Path,
) -> None:
    metadata_path, semantic_path = _write_inputs(tmp_path)
    provider = FakeProvider(fail_generation=True)

    report = AIRuleGenerationService(
        provider=provider,
        artifact_store=ArtifactStore(artifact_dir=tmp_path / "artifacts"),
    ).generate_suggestions(
        metadata_path=metadata_path,
        semantic_report_path=semantic_path,
    )

    assert report.suggestions == []
    assert any("provider failed" in warning for warning in report.warnings)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    metadata_path = tmp_path / "metadata.json"
    semantic_path = tmp_path / "semantic_columns.json"
    metadata = DatasetMetadata(
        source_file="sample.csv",
        file_format="csv",
        row_count=10,
        column_count=1,
        duplicate_count=0,
        columns=[
            ColumnProfile(
                name="title",
                data_type="object",
                null_count=0,
                null_percentage=0.0,
                unique_value_count=2,
                top_values=[
                    TopValue(value="Machine Learning Engineer", count=3),
                    TopValue(value="ML Engineer", count=2),
                ],
                sample_values=[
                    "Machine Learning Engineer",
                    "ML Engineer",
                ],
            )
        ],
    )
    semantic_report = SemanticDetectionReport(
        source_file="sample.csv",
        column_count=1,
        columns=[
            SemanticTag(
                column_name="title",
                semantic_type="JOB_TITLE",
                confidence=0.9,
                detector_name="test",
            )
        ],
    )
    metadata_path.write_text(metadata.model_dump_json(), encoding="utf-8")
    semantic_path.write_text(semantic_report.model_dump_json(), encoding="utf-8")
    return metadata_path, semantic_path
