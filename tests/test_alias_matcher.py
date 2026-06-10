from __future__ import annotations

from core.matching.matcher import AliasMatcher
from models.knowledge import KnowledgeEntry


def test_alias_matcher_matches_exact_aliases() -> None:
    entry = KnowledgeEntry(
        canonical_value="AI Engineer",
        match_type="alias",
        aliases=["ML Engineer", "Machine Learning Engineer"],
        outputs={"job_family": "AI"},
    )
    matcher = AliasMatcher()

    assert matcher.matches("ML Engineer", entry)
    assert matcher.matches(" ai engineer ", entry)
    assert not matcher.matches("Senior ML Engineer", entry)
