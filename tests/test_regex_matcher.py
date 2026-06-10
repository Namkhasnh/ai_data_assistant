from __future__ import annotations

from core.matching.matcher import RegexMatcher
from models.knowledge import KnowledgeEntry


def test_regex_matcher_uses_patterns_only() -> None:
    entry = KnowledgeEntry(
        canonical_value="AI Leadership",
        match_type="regex",
        patterns=[r"\bAI\b.*\b(Lead|Leader|Manager)\b"],
        outputs={"job_family": "AI"},
    )
    matcher = RegexMatcher()

    assert matcher.matches("AI Team Leader", entry)
    assert not matcher.matches("AI Engineer", entry)
    assert not matcher.matches("AI Leadership", entry)


def test_regex_matcher_ignores_invalid_patterns() -> None:
    entry = KnowledgeEntry(
        canonical_value="Invalid",
        match_type="regex",
        patterns=["["],
        outputs={"value": "ignored"},
    )

    assert not RegexMatcher().matches("anything", entry)
