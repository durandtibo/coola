from __future__ import annotations

import pytest

from coola.summary import SetSummarizer, SummaryConfig

CONFIG = SummaryConfig()


class CustomSet(set): ...


class BadLen(set):
    def __len__(self) -> int:
        msg = "broken"
        raise RuntimeError(msg)


###################################
#     Tests for SetSummarizer     #
###################################


def test_set_summarizer_summarize_set() -> None:
    node = SetSummarizer().summarize({3, 1, 2}, CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (
        "<class 'set'> (length=3)",
        [(0, 1), (1, 2), (2, 3)],
        3,
    )


def test_set_summarizer_summarize_frozenset() -> None:
    node = SetSummarizer().summarize(frozenset({2, 1}), CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (
        "<class 'frozenset'> (length=2)",
        [(0, 1), (1, 2)],
        2,
    )


def test_set_summarizer_summarize_empty() -> None:
    node = SetSummarizer().summarize(set(), CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == ("<class 'set'> (length=0)", [], 0)


def test_set_summarizer_summarize_single_item() -> None:
    node = SetSummarizer().summarize({1}, CONFIG, compact=False)
    assert list(node.children) == [(0, 1)]


def test_set_summarizer_summarize_shows_the_smallest_items() -> None:
    node = SetSummarizer().summarize(set(range(10)), SummaryConfig(max_items=3), compact=False)
    assert (list(node.children), node.total) == ([(0, 0), (1, 1), (2, 2)], 10)


def test_set_summarizer_summarize_max_items_none_shows_all_items() -> None:
    node = SetSummarizer().summarize(set(range(10)), SummaryConfig(max_items=None), compact=False)
    assert len(list(node.children)) == 10


def test_set_summarizer_summarize_max_items_zero_shows_no_item() -> None:
    node = SetSummarizer().summarize({1, 2}, SummaryConfig(max_items=0), compact=False)
    assert (list(node.children), node.total) == ([], 2)


def test_set_summarizer_summarize_strings_are_sorted() -> None:
    node = SetSummarizer().summarize({"b", "c", "a"}, CONFIG, compact=False)
    assert list(node.children) == [(0, "a"), (1, "b"), (2, "c")]


def test_set_summarizer_summarize_mixed_types_are_not_compared() -> None:
    node = SetSummarizer().summarize({1, "a"}, SummaryConfig(max_items=1), compact=False)
    assert len(list(node.children)) == 1


def test_set_summarizer_summarize_subclass_keeps_its_type() -> None:
    node = SetSummarizer().summarize(CustomSet({1}), CONFIG, compact=False)
    assert node.header == f"{CustomSet} (length=1)"


def test_set_summarizer_summarize_dict_keys() -> None:
    node = SetSummarizer().summarize({2: "b", 1: "a"}.keys(), CONFIG, compact=False)
    assert list(node.children) == [(0, 1), (1, 2)]


def test_set_summarizer_summarize_does_not_modify_the_data() -> None:
    data = {3, 1, 2}
    list(SetSummarizer().summarize(data, CONFIG, compact=False).children)
    assert data == {1, 2, 3}


def test_set_summarizer_summarize_failing_len_raises() -> None:
    with pytest.raises(RuntimeError, match="broken"):
        SetSummarizer().summarize(BadLen(), CONFIG, compact=False)
