from __future__ import annotations

import pytest

from coola.utils.string import char_diff_summary, remove_empty_lines, truncate_str

##########################################
#     Tests for char_diff_summary        #
##########################################


def test_char_diff_summary_returns_string() -> None:
    assert isinstance(char_diff_summary("hello", "hi"), str)


def test_char_diff_summary_shrink() -> None:
    assert char_diff_summary("<p>Hello</p>", "Hello") == "12 -> 5 chars (-7 chars, -58.3%)."


def test_char_diff_summary_grow() -> None:
    assert char_diff_summary("Hi", "Hello world") == "2 -> 11 chars (+9 chars, +450.0%)."


def test_char_diff_summary_no_change() -> None:
    assert char_diff_summary("hello", "world") == "5 -> 5 chars (+0 chars, +0.0%)."


def test_char_diff_summary_empty_before() -> None:
    assert char_diff_summary("", "hello") == "0 -> 5 chars (+5 chars, +0.0%)."


def test_char_diff_summary_empty_after() -> None:
    assert char_diff_summary("hello", "") == "5 -> 0 chars (-5 chars, -100.0%)."


def test_char_diff_summary_both_empty() -> None:
    assert char_diff_summary("", "") == "0 -> 0 chars (+0 chars, +0.0%)."


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        pytest.param(
            "a" * 1000,
            "a" * 500,
            "1,000 -> 500 chars (-500 chars, -50.0%).",
            id="thousands-separator",
        ),
        pytest.param("ab", "abcd", "2 -> 4 chars (+2 chars, +100.0%).", id="double"),
        pytest.param("abcd", "ab", "4 -> 2 chars (-2 chars, -50.0%).", id="half"),
        pytest.param("abc", "ab", "3 -> 2 chars (-1 chars, -33.3%).", id="one-char-removed"),
    ],
)
def test_char_diff_summary_values(before: str, after: str, expected: str) -> None:
    assert char_diff_summary(before, after) == expected


########################################
#     Tests for remove_empty_lines     #
########################################


def test_remove_empty_lines_removes_empty_lines() -> None:
    assert remove_empty_lines("Hello\n\nWorld") == "Hello\nWorld"


def test_remove_empty_lines_removes_multiple_consecutive_empty_lines() -> None:
    assert remove_empty_lines("Hello\n\n\n\nWorld") == "Hello\nWorld"


def test_remove_empty_lines_removes_whitespace_only_lines() -> None:
    assert remove_empty_lines("Hello\n   \nWorld") == "Hello\nWorld"


def test_remove_empty_lines_removes_tab_only_lines() -> None:
    assert remove_empty_lines("Hello\n\t\nWorld") == "Hello\nWorld"


def test_remove_empty_lines_no_empty_lines() -> None:
    assert remove_empty_lines("Hello\nWorld") == "Hello\nWorld"


def test_remove_empty_lines_all_empty_lines() -> None:
    assert remove_empty_lines("\n\n\n") == ""


def test_remove_empty_lines_empty_string() -> None:
    assert remove_empty_lines("") == ""


def test_remove_empty_lines_single_line_no_newline() -> None:
    assert remove_empty_lines("Hello") == "Hello"


def test_remove_empty_lines_leading_empty_lines() -> None:
    assert remove_empty_lines("\n\nHello") == "Hello"


def test_remove_empty_lines_trailing_empty_lines() -> None:
    assert remove_empty_lines("Hello\n\n") == "Hello"


def test_remove_empty_lines_preserves_internal_whitespace() -> None:
    assert remove_empty_lines("Hello   World\n\nFoo") == "Hello   World\nFoo"


def test_remove_empty_lines_multiline_with_mixed_empty_lines() -> None:
    assert remove_empty_lines("Hello\n\nWorld\n\n\nFoo") == "Hello\nWorld\nFoo"


##################################
#     Tests for truncate_str     #
##################################


def test_truncate_str_short_string() -> None:
    assert truncate_str("hello") == "hello"


def test_truncate_str_exact_length() -> None:
    assert truncate_str("hello", max_len=5) == "hello"


def test_truncate_str_empty_string() -> None:
    assert truncate_str("") == ""


def test_truncate_str_truncated() -> None:
    assert truncate_str("hello world", max_len=8) == "hello..."


def test_truncate_str_truncated_to_suffix_length() -> None:
    assert truncate_str("hello world", max_len=3) == "..."


def test_truncate_str_truncated_single_character() -> None:
    assert truncate_str("hello world", max_len=4) == "h..."


def test_truncate_str_custom_suffix() -> None:
    assert truncate_str("hello world", max_len=8, suffix="…") == "hello w…"


def test_truncate_str_empty_suffix() -> None:
    assert truncate_str("hello world", max_len=5, suffix="") == "hello"


def test_truncate_str_empty_suffix_exact_length() -> None:
    assert truncate_str("hello", max_len=5, suffix="") == "hello"


def test_truncate_str_max_len_less_than_suffix() -> None:
    with pytest.raises(
        ValueError, match=r"max_len .* must be greater than or equal to the suffix length"
    ):
        truncate_str("hello world", max_len=2)


def test_truncate_str_max_len_equal_zero_empty_suffix() -> None:
    assert truncate_str("hello", max_len=0, suffix="") == ""
