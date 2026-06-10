from __future__ import annotations

import pytest

from core.matching.matcher import (
    ContainsMatcher,
    ExactMatcher,
    MatcherError,
    MatcherRegistry,
)


def test_matcher_registry_dispatches_registered_matchers() -> None:
    registry = MatcherRegistry()
    registry.register("exact", ExactMatcher)

    matcher = registry.create("exact")

    assert isinstance(matcher, ExactMatcher)
    assert registry.available_matchers() == ["exact"]


def test_default_matcher_registry_exposes_core_matchers() -> None:
    registry = MatcherRegistry.default()

    assert registry.available_matchers() == [
        "alias",
        "contains",
        "exact",
        "regex",
    ]
    assert isinstance(registry.create("contains"), ContainsMatcher)


def test_matcher_registry_rejects_unknown_matcher() -> None:
    with pytest.raises(MatcherError, match="Unsupported matcher"):
        MatcherRegistry.default().create("prefix")
