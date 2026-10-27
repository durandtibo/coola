from __future__ import annotations

from collections import ChainMap, Counter, OrderedDict, UserDict
from types import MappingProxyType

from coola.summary import MappingSummarizer, SummaryConfig

CONFIG = SummaryConfig()


class CustomDict(dict): ...


#######################################
#     Tests for MappingSummarizer     #
#######################################


def test_mapping_summarizer_summarize_dict() -> None:
    node = MappingSummarizer().summarize({"a": 1, "b": [2]}, CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (
        "<class 'dict'> (length=2)",
        [("a", 1), ("b", [2])],
        2,
    )


def test_mapping_summarizer_summarize_empty() -> None:
    node = MappingSummarizer().summarize({}, CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == ("<class 'dict'> (length=0)", [], 0)


def test_mapping_summarizer_summarize_single_item() -> None:
    node = MappingSummarizer().summarize({"a": 1}, CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (
        "<class 'dict'> (length=1)",
        [("a", 1)],
        1,
    )


def test_mapping_summarizer_summarize_keeps_the_order_of_the_mapping() -> None:
    node = MappingSummarizer().summarize({"b": 1, "a": 2}, CONFIG, compact=False)
    assert list(node.children) == [("b", 1), ("a", 2)]


def test_mapping_summarizer_summarize_non_string_keys() -> None:
    node = MappingSummarizer().summarize({1: "a", (2, 3): "b"}, CONFIG, compact=False)
    assert list(node.children) == [(1, "a"), ((2, 3), "b")]


def test_mapping_summarizer_summarize_ordered_dict() -> None:
    node = MappingSummarizer().summarize(OrderedDict(a=1), CONFIG, compact=False)
    assert node.header == "<class 'collections.OrderedDict'> (length=1)"


def test_mapping_summarizer_summarize_counter() -> None:
    node = MappingSummarizer().summarize(Counter("aab"), CONFIG, compact=False)
    assert list(node.children) == [("a", 2), ("b", 1)]


def test_mapping_summarizer_summarize_user_dict() -> None:
    node = MappingSummarizer().summarize(UserDict(a=1), CONFIG, compact=False)
    assert list(node.children) == [("a", 1)]


def test_mapping_summarizer_summarize_chain_map() -> None:
    node = MappingSummarizer().summarize(ChainMap({"a": 1}, {"b": 2}), CONFIG, compact=False)
    assert node.total == 2


def test_mapping_summarizer_summarize_mapping_proxy() -> None:
    node = MappingSummarizer().summarize(MappingProxyType({"a": 1}), CONFIG, compact=False)
    assert (node.header, list(node.children), node.total) == (
        "<class 'mappingproxy'> (length=1)",
        [("a", 1)],
        1,
    )


def test_mapping_summarizer_summarize_subclass_keeps_its_type() -> None:
    node = MappingSummarizer().summarize(CustomDict(a=1), CONFIG, compact=False)
    assert node.header == f"{CustomDict} (length=1)"


def test_mapping_summarizer_summarize_children_are_lazy() -> None:
    data = {i: i for i in range(100)}
    node = MappingSummarizer().summarize(data, CONFIG, compact=False)
    assert next(iter(node.children)) == (0, 0)


def test_mapping_summarizer_summarize_does_not_modify_the_data() -> None:
    data = {"a": [1]}
    list(MappingSummarizer().summarize(data, CONFIG, compact=False).children)
    assert data == {"a": [1]}
