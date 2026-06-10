from __future__ import annotations

from core.matching.matcher import ContainsMatcher
from models.knowledge import KnowledgeEntry


def test_contains_matcher_matches_canonical_value_or_alias_inside_source() -> None:
    entry = KnowledgeEntry(
        canonical_value="Data Analyst",
        match_type="contains",
        aliases=["Chuyên viên phân tích dữ liệu"],
        outputs={"job_family": "Analytics"},
    )
    matcher = ContainsMatcher()

    assert matcher.matches("Senior Data Analyst", entry)
    assert matcher.matches("Tuyển Chuyên viên phân tích dữ liệu", entry)
    assert not matcher.matches("Data Engineer", entry)
