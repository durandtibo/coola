from __future__ import annotations

import pytest

from coola.display.rich import print_markdown, print_pretty
from coola.display.rich.print_ import _truncate

###############################
#     Tests for _truncate     #
###############################


def test_truncate_shorter_than_max_length_returns_unchanged() -> None:
    assert _truncate("Hello world", 100) == ("Hello world", False)


def test_truncate_exact_max_length_returns_unchanged() -> None:
    content = "a" * 50
    assert _truncate(content, 50) == (content, False)


def test_truncate_longer_than_max_length_is_truncated() -> None:
    content, truncated = _truncate("Hello wonderful world", 10)
    assert truncated is True
    assert len(content) <= 10


def test_truncate_breaks_on_word_boundary() -> None:
    content, truncated = _truncate("Hello wonderful world", 12)
    assert truncated is True
    assert not content.endswith(" ")
    assert "wonderfu" not in content  # word wasn't cut mid-way when a boundary was close


def test_truncate_empty_string() -> None:
    assert _truncate("", 10) == ("", False)


def test_truncate_snaps_to_word_boundary_when_close_to_limit() -> None:
    """last_space (15) is within 80% of max_length (18), so the cut
    should snap back to the word boundary, exercising the `cut =
    cut[:last_space]` branch."""
    content, truncated = _truncate("Hello wonderful world", 18)
    assert content == "Hello wonderful"
    assert truncated is True


def test_truncate_does_not_snap_when_word_boundary_too_far_back() -> None:
    """last_space (5) is below 80% of max_length (12), so the cut is
    kept as-is without snapping back to the word boundary."""
    content, truncated = _truncate("Hello wonderful world", 12)
    assert content == "Hello wonder"
    assert truncated is True


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


def test_print_markdown_max_length_none_renders_full() -> None:
    assert print_markdown("word " * 200, max_length=None) is None


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
