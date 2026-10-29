from __future__ import annotations

import pytest

from coola.display.rich import print_markdown, print_pretty
from coola.display.rich.print_ import _close_code_fence, _truncate_markdown
from coola.testing.marks import rich_available
from coola.utils.imports import is_rich_available

if is_rich_available():
    from io import StringIO

    from rich.console import Console
else:
    StringIO = Console = object

pytestmark = rich_available

#########################################
#     Tests for _close_code_fence     #
#########################################


def test_close_code_fence_without_fence_is_unchanged() -> None:
    assert _close_code_fence("plain text") == "plain text"


def test_close_code_fence_closed_fence_is_unchanged() -> None:
    assert _close_code_fence("```\nx = 1\n```") == "```\nx = 1\n```"


def test_close_code_fence_open_backtick_fence_is_closed() -> None:
    assert _close_code_fence("```python\nx = 1") == "```python\nx = 1\n```"


def test_close_code_fence_open_tilde_fence_is_closed() -> None:
    assert _close_code_fence("~~~\nx = 1") == "~~~\nx = 1\n~~~"


def test_close_code_fence_tilde_line_does_not_close_backtick_fence() -> None:
    assert _close_code_fence("```\n~~~") == "```\n~~~\n```"


def test_close_code_fence_indented_fence_is_closed() -> None:
    assert _close_code_fence("  ```\nx") == "  ```\nx\n```"


def test_close_code_fence_second_block_open_is_closed() -> None:
    assert _close_code_fence("```\na\n```\n\n```\nb") == "```\na\n```\n\n```\nb\n```"


def test_close_code_fence_empty_string() -> None:
    assert _close_code_fence("") == ""


###########################################
#     Tests for _truncate_markdown        #
###########################################


def test_truncate_markdown_shorter_than_max_length_is_unchanged() -> None:
    assert _truncate_markdown("Hello world", 100) == "Hello world"


def test_truncate_markdown_exact_max_length_is_unchanged() -> None:
    assert _truncate_markdown("a" * 50, 50) == "a" * 50


def test_truncate_markdown_empty_string() -> None:
    assert _truncate_markdown("", 10) == ""


def test_truncate_markdown_cuts_at_paragraph_break() -> None:
    # "first paragraph" is 15 characters, the 12 following ones are not shown.
    assert (
        _truncate_markdown("first paragraph\n\nsecond\nthird", 25)
        == "first paragraph\n\n*… (14 more characters)*"
    )


def test_truncate_markdown_prefers_paragraph_break_to_line_break() -> None:
    assert (
        _truncate_markdown("aaaaaaaaaa\n\nbbb\ncccccc", 18)
        == "aaaaaaaaaa\n\n*… (12 more characters)*"
    )


def test_truncate_markdown_cuts_at_line_break() -> None:
    assert (
        _truncate_markdown("line one\nline two\nline three", 20)
        == "line one\nline two\n\n*… (11 more characters)*"
    )


def test_truncate_markdown_cuts_at_space() -> None:
    assert (
        _truncate_markdown("Hello wonderful world", 18)
        == "Hello wonderful\n\n*… (6 more characters)*"
    )


def test_truncate_markdown_paragraph_break_keeping_less_than_half_is_skipped() -> None:
    assert (
        _truncate_markdown("aaaaaaaa\n\nbbb\ncccccc", 17)
        == "aaaaaaaa\n\nbbb\n\n*… (7 more characters)*"
    )


def test_truncate_markdown_hard_cut_when_break_keeps_less_than_half() -> None:
    assert _truncate_markdown("ab cdefghijklmnop", 12) == "ab cdefghijk\n\n*… (5 more characters)*"


def test_truncate_markdown_break_keeping_exactly_half_is_used() -> None:
    assert _truncate_markdown("abcde fghijkl", 10) == "abcde\n\n*… (8 more characters)*"


def test_truncate_markdown_closes_open_code_fence() -> None:
    assert (
        _truncate_markdown("```\nx = 1\ny = 2\nz = 3\n```", 14)
        == "```\nx = 1\n```\n\n*… (16 more characters)*"
    )


def test_truncate_markdown_leaves_closed_code_fence_alone() -> None:
    assert (
        _truncate_markdown("```\nx\n```\n\nsome text after", 12)
        == "```\nx\n```\n\n*… (17 more characters)*"
    )


def test_truncate_markdown_number_of_characters_has_thousands_separator() -> None:
    assert _truncate_markdown("a" * 1500, 100).endswith("*… (1,400 more characters)*")


def test_truncate_markdown_max_length_one() -> None:
    assert _truncate_markdown("abc", 1) == "a\n\n*… (2 more characters)*"


####################################
#     Tests for print_markdown     #
####################################


def test_print_markdown_returns_none() -> None:
    assert print_markdown("**hello**") is None


@pytest.mark.parametrize(
    ("msg", "title"),
    [
        pytest.param("**hello**", None, id="no-title"),
        pytest.param("**hello**", "Demo", id="with-title"),
        pytest.param("", None, id="empty-string"),
        pytest.param("# H1\n\nParagraph.", "Report", id="multiline-with-title"),
    ],
)
def test_print_markdown_renders_panel(msg: str, title: str | None) -> None:
    print_markdown(msg, title=title)


@pytest.mark.parametrize("title_align", ["left", "center", "right"])
def test_print_markdown_title_align(title_align: str) -> None:
    print_markdown("**hello**", title="Demo", title_align=title_align)


@pytest.mark.parametrize("box", [True, False])
def test_print_markdown_box(box: bool) -> None:
    print_markdown("**hello**", box=box)


def test_print_markdown_max_length_no_truncation() -> None:
    assert print_markdown("**hello**", max_length=500) is None


def test_print_markdown_max_length_truncates() -> None:
    assert print_markdown("word " * 200, max_length=50) is None


@pytest.mark.parametrize("max_length", [0, -1])
def test_print_markdown_max_length_lower_than_one_raises(max_length: int) -> None:
    with pytest.raises(ValueError, match=r"max_length"):
        print_markdown("**hello**", max_length=max_length)


def test_print_markdown_max_length_none_renders_full() -> None:
    assert print_markdown("word " * 200, max_length=None) is None


def test_print_markdown_max_length_truncates_output() -> None:
    console = Console(file=StringIO(), width=24)
    print_markdown("alpha beta\n\ngamma delta", panel=False, max_length=12, console=console)
    assert console.file.getvalue() == "alpha beta              \n\n… (13 more characters)  \n"


@pytest.mark.parametrize("panel", [True, False])
def test_print_markdown_panel(panel: bool) -> None:
    assert print_markdown("**hello**", title="Demo", box=False, panel=panel) is None


##################################
#     Tests for print_pretty     #
##################################


def test_print_pretty_returns_none() -> None:
    assert print_pretty({"key": "value"}) is None


@pytest.mark.parametrize(
    ("data", "title"),
    [
        pytest.param({"key": "value"}, None, id="dict-no-title"),
        pytest.param({"key": "value"}, "Demo", id="dict-with-title"),
        pytest.param([], None, id="empty-list"),
        pytest.param([1, 2, 3], "Numbers", id="list-with-title"),
        pytest.param("hello", None, id="string"),
        pytest.param(42, None, id="integer"),
        pytest.param(None, None, id="none"),
        pytest.param({}, "Empty", id="empty-dict-with-title"),
    ],
)
def test_print_pretty_renders_panel(data: object, title: str | None) -> None:
    print_pretty(data, title=title)


@pytest.mark.parametrize("title_align", ["left", "center", "right"])
def test_print_pretty_title_align(title_align: str) -> None:
    print_pretty({"key": "value"}, title="Demo", title_align=title_align)


@pytest.mark.parametrize("box", [True, False])
def test_print_pretty_box(box: bool) -> None:
    print_pretty({"key": "value"}, box=box)


@pytest.mark.parametrize("panel", [True, False])
def test_print_pretty_panel(panel: bool) -> None:
    assert print_pretty({"key": "value"}, title="Demo", box=False, panel=panel) is None
