from __future__ import annotations

import pytest

from coola.utils.string import (
    char_diff_summary,
    count_lines,
    remove_empty_lines,
    truncate_str,
)

##########################################
#     Tests for char_diff_summary        #
##########################################


def test_char_diff_summary_returns_string() -> None:
    assert isinstance(char_diff_summary("hello", "hi"), str)


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        pytest.param("<p>Hello</p>", "Hello", "12 -> 5 chars (-7 chars, -58.3%).", id="shrink"),
        pytest.param("Hi", "Hello world", "2 -> 11 chars (+9 chars, +450.0%).", id="grow"),
        pytest.param("hello", "world", "5 -> 5 chars (+0 chars, +0.0%).", id="no-change"),
        pytest.param("", "hello", "0 -> 5 chars (+5 chars, +0.0%).", id="empty-before"),
        pytest.param("hello", "", "5 -> 0 chars (-5 chars, -100.0%).", id="empty-after"),
        pytest.param("", "", "0 -> 0 chars (+0 chars, +0.0%).", id="both-empty"),
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


#################################
#     Tests for count_lines     #
#################################


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("", 0, id="empty-string"),
        pytest.param("Hello", 1, id="single-line"),
        pytest.param("Hello\nWorld", 2, id="two-lines"),
        pytest.param("Hello\nWorld\nFoo", 3, id="three-lines"),
        pytest.param("Hello\n", 1, id="trailing-newline"),
        pytest.param("\nHello", 2, id="leading-newline"),
        pytest.param("Hello\n\n\nWorld", 4, id="multiple-consecutive-newlines"),
        pytest.param("\n\n\n", 3, id="only-newlines"),
        pytest.param("Hello\r\nWorld", 2, id="windows-line-endings"),
        pytest.param("Hello\nWorld\r\nFoo", 3, id="mixed-line-endings"),
    ],
)
def test_count_lines(text: str, expected: int) -> None:
    assert count_lines(text) == expected


########################################
#     Tests for remove_empty_lines     #
########################################


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("Hello\n\nWorld", "Hello\nWorld", id="single-empty-line"),
        pytest.param("Hello\n\n\n\nWorld", "Hello\nWorld", id="multiple-consecutive-empty-lines"),
        pytest.param("Hello\n   \nWorld", "Hello\nWorld", id="whitespace-only-line"),
        pytest.param("Hello\n\t\nWorld", "Hello\nWorld", id="tab-only-line"),
        pytest.param("Hello\nWorld", "Hello\nWorld", id="no-empty-lines"),
        pytest.param("\n\n\n", "", id="all-empty-lines"),
        pytest.param("", "", id="empty-string"),
        pytest.param("Hello", "Hello", id="single-line-no-newline"),
        pytest.param("\n\nHello", "Hello", id="leading-empty-lines"),
        pytest.param("Hello\n\n", "Hello", id="trailing-empty-lines"),
        pytest.param(
            "Hello   World\n\nFoo", "Hello   World\nFoo", id="preserves-internal-whitespace"
        ),
        pytest.param("Hello\n\nWorld\n\n\nFoo", "Hello\nWorld\nFoo", id="mixed-empty-lines"),
    ],
)
def test_remove_empty_lines(text: str, expected: str) -> None:
    assert remove_empty_lines(text) == expected


##################################
#     Tests for truncate_str     #
##################################


@pytest.mark.parametrize(
    ("text", "kwargs", "expected"),
    [
        pytest.param("hello", {}, "hello", id="short-string"),
        pytest.param("hello", {"max_len": 5}, "hello", id="exact-length"),
        pytest.param("", {}, "", id="empty-string"),
        pytest.param("hello world", {"max_len": 8}, "hello...", id="truncated"),
        pytest.param("hello world", {"max_len": 3}, "...", id="truncated-to-suffix-length"),
        pytest.param("hello world", {"max_len": 4}, "h...", id="truncated-single-character"),
        pytest.param("hello world", {"max_len": 8, "suffix": "…"}, "hello w…", id="custom-suffix"),
        pytest.param("hello world", {"max_len": 5, "suffix": ""}, "hello", id="empty-suffix"),
        pytest.param(
            "hello", {"max_len": 5, "suffix": ""}, "hello", id="empty-suffix-exact-length"
        ),
        pytest.param("hello", {"max_len": 0, "suffix": ""}, "", id="max-len-zero-empty-suffix"),
    ],
)
def test_truncate_str(text: str, kwargs: dict, expected: str) -> None:
    assert truncate_str(text, **kwargs) == expected


def test_truncate_str_max_len_less_than_suffix() -> None:
    with pytest.raises(
        ValueError, match=r"max_len .* must be greater than or equal to the suffix length"
    ):
        truncate_str("hello world", max_len=2)
