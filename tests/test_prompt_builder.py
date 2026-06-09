from __future__ import annotations

from pathlib import Path

from core.ai.prompt_builder import PromptBuilder
from models.column_profile import ColumnProfile, TopValue
from models.dataset import DatasetMetadata
from models.semantic_tag import SemanticDetectionReport, SemanticTag


def test_prompt_builder_uses_compact_metadata_and_semantic_context(tmp_path: Path) -> None:
    template_path = tmp_path / "rule_generation.txt"
    template_path.write_text("Context:\n{{CONTEXT_JSON}}", encoding="utf-8")
    metadata = DatasetMetadata(
        source_file="sample.csv",
        file_format="csv",
        row_count=100,
        column_count=1,
        duplicate_count=0,
        columns=[
            ColumnProfile(
                name="title",
                data_type="object",
                null_count=0,
                null_percentage=0.0,
                unique_value_count=3,
                top_values=[
                    TopValue(value="Machine Learning Engineer", count=4),
                    TopValue(value="ML Engineer", count=2),
                    TopValue(value="AI Engineer", count=1),
                ],
                sample_values=[
                    "Machine Learning Engineer",
                    "ML Engineer",
                    "AI Engineer",
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

    prompt = PromptBuilder(
        template_path=template_path,
        max_sample_values=2,
        max_top_values=2,
    ).build_rule_generation_prompt(metadata, semantic_report)

    assert "sample.csv" in prompt
    assert "JOB_TITLE" in prompt
    assert "Machine Learning Engineer" in prompt
    assert "ML Engineer" in prompt
    assert "AI Engineer" not in prompt
