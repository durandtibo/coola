from __future__ import annotations

from collections import UserList, deque
from typing import NamedTuple

import pytest

from coola.summary import SequenceSummarizer, SummaryConfig

CONFIG = SummaryConfig()


class Point(NamedTuple):
    x: int
    y: int


class CustomList(list): ...


class BadLen(list):
    def __len__(self) -> int:
        msg = "broken"
        raise RuntimeError(msg)


########################################
#     Tests for SequenceSummarizer     #
########################################


def test_sequence_summarizer_summarize_list() -> None:
    node = SequenceSummarizer().summarize([1, "a", 3.5], CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (
        "<class 'list'> (length=3)",
        [(0, 1), (1, "a"), (2, 3.5)],
        3,
    )


def test_sequence_summarizer_summarize_tuple() -> None:
    node = SequenceSummarizer().summarize((1, 2), CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (
        "<class 'tuple'> (length=2)",
        [(0, 1), (1, 2)],
        2,
    )


def test_sequence_summarizer_summarize_deque() -> None:
    node = SequenceSummarizer().summarize(deque([1, 2]), CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (
        "<class 'collections.deque'> (length=2)",
        [(0, 1), (1, 2)],
        2,
    )


def test_sequence_summarizer_summarize_user_list() -> None:
    node = SequenceSummarizer().summarize(UserList([1]), CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (
        "<class 'collections.UserList'> (length=1)",
        [(0, 1)],
        1,
    )


def test_sequence_summarizer_summarize_range() -> None:
    node = SequenceSummarizer().summarize(range(2), CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (
        "<class 'range'> (length=2)",
        [(0, 0), (1, 1)],
        2,
    )


def test_sequence_summarizer_summarize_empty() -> None:
    node = SequenceSummarizer().summarize([], CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == ("<class 'list'> (length=0)", [], 0)


def test_sequence_summarizer_summarize_single_item() -> None:
    node = SequenceSummarizer().summarize([1], CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (
        "<class 'list'> (length=1)",
        [(0, 1)],
        1,
    )


def test_sequence_summarizer_summarize_duplicate_items() -> None:
    assert list(SequenceSummarizer().summarize([1, 1], CONFIG, compact=False).children) == [
        (0, 1),
        (1, 1),
    ]


def test_sequence_summarizer_summarize_subclass_keeps_its_type() -> None:
    node = SequenceSummarizer().summarize(CustomList([1]), CONFIG, compact=False)
    assert node.header == f"{CustomList} (length=1)"


def test_sequence_summarizer_summarize_named_tuple_uses_the_field_names() -> None:
    node = SequenceSummarizer().summarize(Point(1, 2), CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (
        f"{Point} (length=2)",
        [("x", 1), ("y", 2)],
        2,
    )


def test_sequence_summarizer_summarize_length_is_formatted() -> None:
    node = SequenceSummarizer().summarize(range(1000), CONFIG, compact=False)
    assert node.header == "<class 'range'> (length=1,000)"


def test_sequence_summarizer_summarize_children_are_lazy() -> None:
    node = SequenceSummarizer().summarize(range(10**12), CONFIG, compact=False)
    assert next(iter(node.children)) == (0, 0)


def test_sequence_summarizer_summarize_compact_is_ignored() -> None:
    summarizer = SequenceSummarizer()
    assert summarizer.summarize([1], CONFIG, compact=True).header == (
        summarizer.summarize([1], CONFIG, compact=False).header
    )


def test_sequence_summarizer_summarize_does_not_modify_the_data() -> None:
    data = [1, [2]]
    list(SequenceSummarizer().summarize(data, CONFIG, compact=False).children)
    assert data == [1, [2]]


def test_sequence_summarizer_summarize_failing_len_raises() -> None:
    with pytest.raises(RuntimeError, match="broken"):
        SequenceSummarizer().summarize(BadLen(), CONFIG, compact=False)
