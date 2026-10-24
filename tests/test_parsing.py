from __future__ import annotations

import pytest

from agentic_assistant.parsing import preprocess_math_expression, safe_parse_json, safe_parse_json_list


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("what is 3 minus 2?", "3 - 2"),
        ("divide 100 by 0", "100 / 0"),
        ("multiply 5 by 3", "5 * 3"),
        ("add 2 to 3", "2 + 3"),
        ("subtract 2 from 10", "10 - 2"),
        ("5 mul by 6", "5 * 6"),
        ("3 x 4", "3 * 4"),
        ("2^10", "2**10"),
        ("what is 10 squared", "10 ** 2"),
        ("100 divided by 5", "100 / 5"),
        ("calculate 25 * 4", "25 * 4"),
        ("what is the square root of 81", "sqrt(81)"),
    ],
)
def test_preprocess_math_expression(text: str, expected: str) -> None:
    assert preprocess_math_expression(text) == expected


def test_safe_parse_json_handles_fences_and_prose() -> None:
    assert safe_parse_json('```json\n{"tool": "calculator"}\n```') == {"tool": "calculator"}
    assert safe_parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert safe_parse_json("not json") == {}
    assert safe_parse_json("") == {}


def test_safe_parse_json_list() -> None:
    assert safe_parse_json_list('```json\n[{"query": "5+3"}]\n```') == [{"query": "5+3"}]
    assert safe_parse_json_list('{"not": "a list"}') is None
