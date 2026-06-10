from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Any

import pandas as pd

from models.knowledge import KnowledgeEntry


class MatcherError(ValueError):
    """Raised when a matcher cannot be created."""


class BaseMatcher(ABC):
    """Base interface for deterministic knowledge matching."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def matches(self, source_value: Any, entry: KnowledgeEntry) -> bool:
        """Return whether the source value matches a knowledge entry."""

    @staticmethod
    def normalize(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).strip().casefold()

    @classmethod
    def normalized_candidates(cls, entry: KnowledgeEntry) -> list[str]:
        candidates = [entry.canonical_value, *entry.aliases]
        return [
            normalized
            for candidate in candidates
            if (normalized := cls.normalize(candidate))
        ]


class ExactMatcher(BaseMatcher):
    """Match when the source value exactly equals the canonical value."""

    def __init__(self) -> None:
        super().__init__(name="exact")

    def matches(self, source_value: Any, entry: KnowledgeEntry) -> bool:
        return self.normalize(source_value) == self.normalize(entry.canonical_value)


class ContainsMatcher(BaseMatcher):
    """Match when the source value contains the canonical value or aliases."""

    def __init__(self) -> None:
        super().__init__(name="contains")

    def matches(self, source_value: Any, entry: KnowledgeEntry) -> bool:
        normalized_source = self.normalize(source_value)
        if not normalized_source:
            return False
        return any(
            candidate in normalized_source
            for candidate in self.normalized_candidates(entry)
        )


class AliasMatcher(BaseMatcher):
    """Match when the source value exactly equals the canonical value or an alias."""

    def __init__(self) -> None:
        super().__init__(name="alias")

    def matches(self, source_value: Any, entry: KnowledgeEntry) -> bool:
        normalized_source = self.normalize(source_value)
        if not normalized_source:
            return False
        return normalized_source in set(self.normalized_candidates(entry))


class RegexMatcher(BaseMatcher):
    """Match when any configured regex pattern matches the source value."""

    def __init__(self) -> None:
        super().__init__(name="regex")

    def matches(self, source_value: Any, entry: KnowledgeEntry) -> bool:
        if pd.isna(source_value):
            return False
        source_text = str(source_value)
        for pattern in entry.patterns:
            try:
                if re.search(pattern, source_text, flags=re.IGNORECASE):
                    return True
            except re.error:
                continue
        return False


class MatcherRegistry:
    """Plugin registry for deterministic knowledge matchers."""

    def __init__(self) -> None:
        self._matchers: dict[str, type[BaseMatcher]] = {}

    def register(self, name: str, matcher_class: type[BaseMatcher]) -> None:
        self._matchers[name.strip().lower()] = matcher_class

    def create(self, name: str) -> BaseMatcher:
        matcher_class = self._matchers.get(name.strip().lower())
        if matcher_class is None:
            raise MatcherError(f"Unsupported matcher: {name}")
        return matcher_class()

    def available_matchers(self) -> list[str]:
        return sorted(self._matchers)

    @classmethod
    def default(cls) -> MatcherRegistry:
        registry = cls()
        registry.register("exact", ExactMatcher)
        registry.register("contains", ContainsMatcher)
        registry.register("alias", AliasMatcher)
        registry.register("regex", RegexMatcher)
        return registry
