from __future__ import annotations

from core.matching.matcher import ExactMatcher
from models.knowledge import KnowledgeEntry


def test_exact_matcher_matches_canonical_value_only() -> None:
    entry = KnowledgeEntry(
        canonical_value="Data Analyst",
        match_type="exact",
        aliases=["Business Data Analyst"],
        outputs={"job_family": "Analytics"},
    )
    matcher = ExactMatcher()

    assert matcher.matches(" data analyst ", entry)
    assert not matcher.matches("Business Data Analyst", entry)
