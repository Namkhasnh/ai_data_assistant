from __future__ import annotations

import pytest

from core.ai.response_parser import AIResponseParseError, ResponseParser


def test_response_parser_parses_valid_json_deterministically() -> None:
    raw_response = """
    {
      "suggestions": [
        {
          "type": "mapping",
          "column": "title",
          "semantic_type": "JOB_TITLE",
          "source_value": "Machine Learning Engineer",
          "target_value": "AI Engineer",
          "confidence": 0.92,
          "reason": "Similar role naming",
          "created_by": "llm"
        }
      ]
    }
    """
    parser = ResponseParser()

    first = parser.parse(raw_response)
    second = parser.parse(raw_response)

    assert first == second
    assert first[0].type == "mapping"
    assert first[0].source_value == "Machine Learning Engineer"


def test_response_parser_rejects_malformed_json() -> None:
    with pytest.raises(AIResponseParseError, match="valid JSON"):
        ResponseParser().parse("{not json")


def test_response_parser_rejects_invalid_shape() -> None:
    with pytest.raises(AIResponseParseError, match="suggestions list"):
        ResponseParser().parse('{"suggestions": {}}')
