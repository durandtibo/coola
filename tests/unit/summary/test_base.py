from __future__ import annotations

import pytest

from coola.summary import BaseSummarizer, Node, SummaryConfig

CONFIG = SummaryConfig()


class FirstSummarizer(BaseSummarizer[object]):
    def summarize(self, data: object, config: SummaryConfig, *, compact: bool) -> str:  # noqa: ARG002
        return "first"


class SecondSummarizer(FirstSummarizer): ...


##########################
#     Tests for Node     #
##########################


def test_node_attributes() -> None:
    node = Node("header", enumerate(["a"]), 1)
    assert (node.header, list(node.children), node.total) == ("header", [(0, "a")], 1)


def test_node_equal() -> None:
    assert Node("header", [(0, "a")], 1) == Node("header", [(0, "a")], 1)


def test_node_different_total() -> None:
    assert Node("header", [], 1) != Node("header", [], 2)


####################################
#     Tests for BaseSummarizer     #
####################################


def test_base_summarizer_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseSummarizer()  # ty: ignore[call-non-callable]


def test_base_summarizer_equal_same_type() -> None:
    assert FirstSummarizer().equal(FirstSummarizer())


def test_base_summarizer_equal_different_type() -> None:
    assert not FirstSummarizer().equal(42)


def test_base_summarizer_equal_child_class() -> None:
    assert not FirstSummarizer().equal(SecondSummarizer())


def test_base_summarizer_repr() -> None:
    assert repr(FirstSummarizer()) == "FirstSummarizer()"


def test_base_summarizer_summarize() -> None:
    assert FirstSummarizer().summarize(1, CONFIG, compact=False) == "first"
