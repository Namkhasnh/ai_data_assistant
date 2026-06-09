from __future__ import annotations

import json
from pathlib import Path

from core.semantic.base_detector import SemanticDetectionInput, SemanticRule
from core.semantic.detector_registry import SemanticDetectorRegistry
from core.semantic.keyword_detector import KeywordSemanticDetector
from core.semantic.regex_detector import RegexSemanticDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_keyword_detector_uses_rules_without_domain_specific_classes() -> None:
    detector = KeywordSemanticDetector(
        rules=[
            SemanticRule(
                semantic_type="PRODUCT_ID",
                keywords=["sku", "product code"],
                confidence=0.8,
            )
        ]
    )

    tag = detector.detect(
        SemanticDetectionInput(
            column_name="sku",
            sample_values=["ABC-123"],
            top_values=["XYZ-999"],
        )
    )

    assert tag is not None
    assert tag.semantic_type == "PRODUCT_ID"
    assert tag.detector_name == "keyword_detector"
    assert tag.confidence >= 0.8
    assert tag.evidence


def test_regex_detector_uses_rules_without_domain_specific_classes() -> None:
    detector = RegexSemanticDetector(
        rules=[
            SemanticRule(
                semantic_type="EMAIL",
                patterns=[r"^[^@\s]+@[^@\s]+\.[^@\s]+$"],
                confidence=0.9,
            )
        ]
    )

    tag = detector.detect(
        SemanticDetectionInput(
            column_name="contact",
            sample_values=["person@example.com"],
            top_values=[],
        )
    )

    assert tag is not None
    assert tag.semantic_type == "EMAIL"
    assert tag.detector_name == "regex_detector"
    assert tag.confidence >= 0.9
    assert tag.evidence


def test_registry_loads_rules_from_knowledge_file(tmp_path: Path) -> None:
    knowledge_path = tmp_path / "semantic_rules.json"
    knowledge_path.write_text(
        json.dumps(
            {
                "version": 1,
                "rules": [
                    {
                        "semantic_type": "CUSTOMER_ID",
                        "keywords": ["customer_id"],
                        "patterns": [],
                        "confidence": 0.75,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    registry = SemanticDetectorRegistry.from_knowledge_file(knowledge_path)
    tag = registry.detect_column(
        SemanticDetectionInput(
            column_name="customer_id",
            sample_values=["C001"],
            top_values=["C002"],
        )
    )

    assert tag.semantic_type == "CUSTOMER_ID"
    assert tag.detector_name == "keyword_detector"


def test_registry_returns_unknown_when_no_detector_matches() -> None:
    registry = SemanticDetectorRegistry(detectors=[])

    tag = registry.detect_column(
        SemanticDetectionInput(
            column_name="unmapped_column",
            sample_values=["value"],
            top_values=[],
        )
    )

    assert tag.semantic_type == "UNKNOWN"
    assert tag.confidence == 0.0
    assert tag.detector_name == "detector_registry"
    assert tag.evidence == []


def test_registry_prefers_column_name_evidence_over_sample_value_noise() -> None:
    registry = SemanticDetectorRegistry.from_knowledge_dir(PROJECT_ROOT / "knowledge")

    tag = registry.detect_column(
        SemanticDetectionInput(
            column_name="title",
            sample_values=["Kế Toán Trưởng - Thu Nhập Từ 30 - 40 Triệu / Tháng"],
            top_values=["Data Engineer"],
        )
    )

    assert tag.semantic_type == "JOB_TITLE"
    assert any("column_name" in evidence for evidence in tag.evidence)


def test_default_knowledge_rules_detect_topcv_columns() -> None:
    registry = SemanticDetectorRegistry.from_knowledge_dir(PROJECT_ROOT / "knowledge")

    tags = registry.detect_many(
        [
            SemanticDetectionInput(column_name="title"),
            SemanticDetectionInput(column_name="salary", sample_values=["30 - 45 triệu"]),
            SemanticDetectionInput(column_name="location", top_values=["Hồ Chí Minh"]),
            SemanticDetectionInput(column_name="experience", sample_values=["2 năm"]),
            SemanticDetectionInput(column_name="company_name"),
        ]
    )

    semantic_types = {tag.column_name: tag.semantic_type for tag in tags}
    assert semantic_types["title"] == "JOB_TITLE"
    assert semantic_types["salary"] == "SALARY"
    assert semantic_types["location"] == "LOCATION"
    assert semantic_types["experience"] == "EXPERIENCE"
    assert semantic_types["company_name"] == "COMPANY"
