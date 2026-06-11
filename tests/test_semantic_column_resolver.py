from __future__ import annotations

import logging
import warnings

from core.semantic_resolver.semantic_column_resolver import SemanticColumnResolver
from models.semantic_column_registry import SemanticColumnRegistry


def test_registry_normalizes_duplicate_columns_and_deterministic_ordering() -> None:
    registry = SemanticColumnRegistry(
        columns_by_semantic_type={
            "JOB_REQUIREMENTS": ["requirements", "requirements", "must_have"],
            "JOB_DESCRIPTION": ["description"],
        }
    )

    assert registry.columns_by_semantic_type == {
        "JOB_DESCRIPTION": ["description"],
        "JOB_REQUIREMENTS": ["must_have", "requirements"],
    }


def test_resolver_get_column_returns_first_deterministic_column() -> None:
    resolver = SemanticColumnResolver(
        SemanticColumnRegistry(
            columns_by_semantic_type={
                "JOB_REQUIREMENTS": ["requirements", "must_have"],
                "JOB_DESCRIPTION": ["description"],
            }
        )
    )

    assert resolver.get_column("JOB_REQUIREMENTS") == "must_have"
    assert resolver.get_column("JOB_DESCRIPTION") == "description"


def test_resolver_get_columns_deduplicates_semantic_type_input() -> None:
    resolver = SemanticColumnResolver(
        SemanticColumnRegistry(
            columns_by_semantic_type={
                "JOB_REQUIREMENTS": ["requirements", "must_have"],
                "JOB_DESCRIPTION": ["description"],
            }
        )
    )

    assert resolver.get_columns(
        [
            "JOB_REQUIREMENTS",
            "JOB_REQUIREMENTS",
            "JOB_DESCRIPTION",
        ]
    ) == ["description", "must_have", "requirements"]


def test_resolver_available_semantic_types_are_sorted() -> None:
    resolver = SemanticColumnResolver(
        SemanticColumnRegistry(
            columns_by_semantic_type={
                "JOB_TITLE": ["title"],
                "JOB_DESCRIPTION": ["description"],
                "JOB_REQUIREMENTS": ["requirements"],
            }
        )
    )

    assert resolver.available_semantic_types() == [
        "JOB_DESCRIPTION",
        "JOB_REQUIREMENTS",
        "JOB_TITLE",
    ]


def test_resolver_missing_semantic_types_are_quiet() -> None:
    resolver = SemanticColumnResolver(
        SemanticColumnRegistry(columns_by_semantic_type={"JOB_TITLE": ["title"]})
    )

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        assert resolver.get_column("BANKING_ABBREVIATION") is None
        assert resolver.get_columns(["BANKING_ABBREVIATION"]) == []
        assert resolver.has_semantic_type("BANKING_ABBREVIATION") is False

    assert captured_warnings == []


def test_resolver_empty_registry_behavior() -> None:
    resolver = SemanticColumnResolver()

    assert resolver.get_column("JOB_TITLE") is None
    assert resolver.get_columns(["JOB_TITLE", "JOB_DESCRIPTION"]) == []
    assert resolver.has_semantic_type("JOB_TITLE") is False
    assert resolver.available_semantic_types() == []


def test_resolver_does_not_log_for_missing_semantic_types(caplog) -> None:
    resolver = SemanticColumnResolver()

    with caplog.at_level(logging.WARNING):
        assert resolver.get_column("UNKNOWN") is None
        assert resolver.get_columns(["UNKNOWN"]) == []

    assert caplog.records == []
