from __future__ import annotations

from coola.utils.string import remove_empty_lines

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
