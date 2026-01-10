from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import pytest

from coola.summary import DefaultSummarizer, MappingSummarizer, SummarizerRegistry


@pytest.fixture
def registry() -> SummarizerRegistry:
    return SummarizerRegistry(
        {object: DefaultSummarizer(), Mapping: MappingSummarizer(), dict: MappingSummarizer()}
    )


#######################################
#     Tests for MappingSummarizer     #
#######################################


def test_mapping_summarizer_init_default() -> None:
    summarizer = MappingSummarizer()
    assert summarizer._max_items == 5
    assert summarizer._num_spaces == 2


def test_mapping_summarizer_init_custom() -> None:
    summarizer = MappingSummarizer(max_items=10, num_spaces=4)
    assert summarizer._max_items == 10
    assert summarizer._num_spaces == 4


def test_mapping_summarizer_init_negative_max_items() -> None:
    summarizer = MappingSummarizer(max_items=-1)
    assert summarizer._max_items == -1


def test_mapping_summarizer_repr() -> None:
    summarizer = MappingSummarizer()
    assert repr(summarizer) == "MappingSummarizer(max_items=5, num_spaces=2)"


def test_mapping_summarizer_repr_large_max_items() -> None:
    summarizer = MappingSummarizer(max_items=1000)
    assert repr(summarizer) == "MappingSummarizer(max_items=1,000, num_spaces=2)"


def test_mapping_summarizer_str() -> None:
    summarizer = MappingSummarizer()
    assert str(summarizer) == "MappingSummarizer(max_items=5, num_spaces=2)"


def test_mapping_summarizer_str_large_max_items() -> None:
    summarizer = MappingSummarizer(max_items=1000)
    assert str(summarizer) == "MappingSummarizer(max_items=1,000, num_spaces=2)"


def test_mapping_summarizer_equal_true() -> None:
    assert MappingSummarizer(max_items=10, num_spaces=4).equal(
        MappingSummarizer(max_items=10, num_spaces=4)
    )


def test_mapping_summarizer_equal_true_default() -> None:
    assert MappingSummarizer().equal(MappingSummarizer())


def test_mapping_summarizer_equal_false_different_max_items() -> None:
    assert not MappingSummarizer(max_items=10, num_spaces=4).equal(
        MappingSummarizer(max_items=5, num_spaces=4)
    )


def test_mapping_summarizer_equal_false_different_num_spaces() -> None:
    assert not MappingSummarizer(max_items=10, num_spaces=4).equal(
        MappingSummarizer(max_items=10, num_spaces=2)
    )


def test_mapping_summarizer_equal_false_different_type() -> None:
    assert not MappingSummarizer().equal(42)


def test_mapping_summarizer_equal_false_different_type_child() -> None:
    class Child(MappingSummarizer): ...

    assert not MappingSummarizer().equal(Child())


def test_mapping_summarizer_equal_false_none() -> None:
    assert not MappingSummarizer().equal(None)


def test_mapping_summarizer_summarize_empty(registry: SummarizerRegistry) -> None:
    result = MappingSummarizer().summarize({}, registry)
    assert result == "<class 'dict'> {}"


def test_mapping_summarizer_summarize_single_item(registry: SummarizerRegistry) -> None:
    result = MappingSummarizer().summarize({"key1": "value1"}, registry)
    assert result == "<class 'dict'> (length=1)\n  (key1): value1"


def test_mapping_summarizer_summarize_multiple_items(registry: SummarizerRegistry) -> None:
    result = MappingSummarizer().summarize({"key1": 1.2, "key2": "abc", "key3": 42}, registry)
    assert result == "<class 'dict'> (length=3)\n  (key1): 1.2\n  (key2): abc\n  (key3): 42"


def test_mapping_summarizer_summarize_max_items_exceeded(registry: SummarizerRegistry) -> None:
    result = MappingSummarizer(max_items=2).summarize(
        {"key1": 1, "key2": 2, "key3": 3, "key4": 4}, registry
    )
    assert result == "<class 'dict'> (length=4)\n  (key1): 1\n  (key2): 2\n  ..."


def test_mapping_summarizer_summarize_max_items_negative(registry: SummarizerRegistry) -> None:
    data = {f"key{i}": i for i in range(10)}
    result = MappingSummarizer(max_items=-1).summarize(data, registry)
    assert result == (
        "<class 'dict'> (length=10)\n"
        "  (key0): 0\n"
        "  (key1): 1\n"
        "  (key2): 2\n"
        "  (key3): 3\n"
        "  (key4): 4\n"
        "  (key5): 5\n"
        "  (key6): 6\n"
        "  (key7): 7\n"
        "  (key8): 8\n"
        "  (key9): 9"
    )


def test_mapping_summarizer_summarize_depth_exceeds_max_depth(registry: SummarizerRegistry) -> None:
    result = MappingSummarizer().summarize({"key1": "value1"}, registry, depth=1, max_depth=1)
    assert result == "{'key1': 'value1'}"


def test_mapping_summarizer_summarize_custom_num_spaces(registry: SummarizerRegistry) -> None:
    result = MappingSummarizer(num_spaces=4).summarize({"key1": "value1"}, registry)
    assert result == "<class 'dict'> (length=1)\n    (key1): value1"


def test_mapping_summarizer_summarize_ordered_dict(registry: SummarizerRegistry) -> None:
    data = OrderedDict([("key1", 1), ("key2", 2), ("key3", 3)])
    result = MappingSummarizer().summarize(data, registry)
    assert (
        result == "<class 'collections.OrderedDict'> (length=3)\n"
        "  (key1): 1\n"
        "  (key2): 2\n"
        "  (key3): 3"
    )


def test_mapping_summarizer_summarize_nested_data(registry: SummarizerRegistry) -> None:
    data = {"key1": {"nested": "value"}, "key2": "simple"}
    result = MappingSummarizer().summarize(data, registry, depth=0, max_depth=2)
    assert result == (
        "<class 'dict'> (length=2)\n"
        "  (key1): <class 'dict'> (length=1)\n"
        "      (nested): value\n"
        "  (key2): <class 'str'> simple"
    )


def test_mapping_summarizer_summarize_nested_data_2(registry: SummarizerRegistry) -> None:
    data = {"key0": {"key0": 0, "key1": 1, "key2": 1}, "key1": {1: 2, 3: 4}}
    result = MappingSummarizer().summarize(data, registry, max_depth=2)
    assert result == (
        "<class 'dict'> (length=2)\n"
        "  (key0): <class 'dict'> (length=3)\n"
        "      (key0): 0\n"
        "      (key1): 1\n"
        "      (key2): 1\n"
        "  (key1): <class 'dict'> (length=2)\n"
        "      (1): 2\n"
        "      (3): 4"
    )


def test_mapping_summarizer_summarize_max_items_equal_length(registry: SummarizerRegistry) -> None:
    result = MappingSummarizer(max_items=3).summarize({"key1": 1, "key2": 2, "key3": 3}, registry)
    assert result == "<class 'dict'> (length=3)\n  (key1): 1\n  (key2): 2\n  (key3): 3"


def test_mapping_summarizer_summarize_max_items_zero(registry: SummarizerRegistry) -> None:
    result = MappingSummarizer(max_items=0).summarize({"key1": 1, "key2": 2, "key3": 3}, registry)
    assert result == "<class 'dict'> (length=3) ..."
