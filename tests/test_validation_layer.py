from __future__ import annotations

from core.ai.validation_layer import ValidationLayer
from models.ai_suggestion import AISuggestion


def test_validation_layer_filters_invalid_suggestions_and_removes_duplicates() -> None:
    valid = AISuggestion(
        type="Mapping",
        column="title",
        semantic_type="JOB_TITLE",
        source_value="Machine Learning Engineer",
        target_value="AI Engineer",
        confidence=0.92,
        reason="Similar role naming",
        created_by="llm",
    )
    duplicate = AISuggestion(
        type="mapping",
        column=" title ",
        semantic_type="JOB_TITLE",
        source_value=" Machine Learning Engineer ",
        target_value=" AI Engineer ",
        confidence=0.95,
        reason="Duplicate suggestion",
        created_by="llm",
    )
    low_confidence = AISuggestion(
        type="mapping",
        source_value="ML",
        target_value="Machine Learning",
        confidence=0.2,
        reason="Too weak",
        created_by="llm",
    )
    invalid_type = AISuggestion(
        type="enrichment",
        confidence=0.9,
        reason="Not allowed in Phase 5A",
        created_by="llm",
    )

    report = ValidationLayer(minimum_confidence=0.8).validate(
        [valid, duplicate, low_confidence, invalid_type],
        provider="test",
        model="fake-model",
    )

    assert len(report.suggestions) == 1
    assert report.suggestions[0].type == "mapping"
    assert report.suggestions[0].column == "title"
    assert report.suggestions[0].suggestion_id is not None
    assert report.suggestions[0].suggestion_id.startswith("ai_suggestion_")
    second_report = ValidationLayer(minimum_confidence=0.8).validate([valid])
    assert report.suggestions[0].suggestion_id == second_report.suggestions[0].suggestion_id
    assert report.duplicate_suggestions_removed == 1
    assert report.rejected_suggestions_count == 2
    assert any("minimum confidence" in warning for warning in report.warnings)
    assert any("unsupported AI suggestion type" in warning for warning in report.warnings)


def test_validation_layer_rejects_invalid_confidence_range() -> None:
    suggestion = AISuggestion(
        type="mapping",
        source_value="A",
        target_value="B",
        confidence=1.2,
        reason="Invalid confidence",
        created_by="llm",
    )

    report = ValidationLayer().validate([suggestion])

    assert report.suggestions == []
    assert report.rejected_suggestions_count == 1
    assert "invalid confidence" in report.warnings[0]
