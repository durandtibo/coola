from __future__ import annotations

from collections.abc import Set as AbstractSet

import pytest

from coola.summary import DefaultSummarizer, SetSummarizer, SummarizerRegistry


@pytest.fixture
def registry() -> SummarizerRegistry:
    return SummarizerRegistry(
        {
            object: DefaultSummarizer(),
            AbstractSet: SetSummarizer(),
            set: SetSummarizer(),
        }
    )


###################################
#     Tests for SetSummarizer     #
###################################


def test_set_summarizer_init_default() -> None:
    summarizer = SetSummarizer()
    assert summarizer._max_items == 5
    assert summarizer._num_spaces == 2


def test_set_summarizer_init_custom() -> None:
    summarizer = SetSummarizer(max_items=10, num_spaces=4)
    assert summarizer._max_items == 10
    assert summarizer._num_spaces == 4


def test_set_summarizer_init_negative_max_items() -> None:
    summarizer = SetSummarizer(max_items=-1)
    assert summarizer._max_items == -1


def test_set_summarizer_repr() -> None:
    summarizer = SetSummarizer()
    assert repr(summarizer) == "SetSummarizer(max_items=5, num_spaces=2)"


def test_set_summarizer_repr_large_max_items() -> None:
    summarizer = SetSummarizer(max_items=1000)
    assert repr(summarizer) == "SetSummarizer(max_items=1,000, num_spaces=2)"


def test_set_summarizer_str() -> None:
    summarizer = SetSummarizer()
    assert str(summarizer) == "SetSummarizer(max_items=5, num_spaces=2)"


def test_set_summarizer_str_large_max_items() -> None:
    summarizer = SetSummarizer(max_items=1000)
    assert str(summarizer) == "SetSummarizer(max_items=1,000, num_spaces=2)"


def test_set_summarizer_equal_true() -> None:
    assert SetSummarizer(max_items=10, num_spaces=4).equal(
        SetSummarizer(max_items=10, num_spaces=4)
    )


def test_set_summarizer_equal_true_default() -> None:
    assert SetSummarizer().equal(SetSummarizer())


def test_set_summarizer_equal_false_different_max_items() -> None:
    assert not SetSummarizer(max_items=10, num_spaces=4).equal(
        SetSummarizer(max_items=5, num_spaces=4)
    )


def test_set_summarizer_equal_false_different_num_spaces() -> None:
    assert not SetSummarizer(max_items=10, num_spaces=4).equal(
        SetSummarizer(max_items=10, num_spaces=2)
    )


def test_set_summarizer_equal_false_different_type() -> None:
    assert not SetSummarizer().equal(42)


def test_set_summarizer_equal_false_different_type_child() -> None:
    class Child(SetSummarizer): ...

    assert not SetSummarizer().equal(Child())


def test_set_summarizer_equal_false_none() -> None:
    assert not SetSummarizer().equal(None)


def test_set_summarizer_empty_set(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summary(set(), registry)
    assert result == "<class 'set'> set()"


def test_set_summarizer_single_item(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summary({1}, registry)
    assert result == "<class 'set'> (length=1)\n  (0): 1"


def test_set_summarizer_multiple_items(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summary({1, 2, 3}, registry)
    # Since sets are unordered, check all possible orderings
    possible_results = [
        "<class 'set'> (length=3)\n  (0): 1\n  (1): 2\n  (2): 3",
        "<class 'set'> (length=3)\n  (0): 1\n  (1): 3\n  (2): 2",
        "<class 'set'> (length=3)\n  (0): 2\n  (1): 1\n  (2): 3",
        "<class 'set'> (length=3)\n  (0): 2\n  (1): 3\n  (2): 1",
        "<class 'set'> (length=3)\n  (0): 3\n  (1): 1\n  (2): 2",
        "<class 'set'> (length=3)\n  (0): 3\n  (1): 2\n  (2): 1",
    ]
    assert result in possible_results


def test_set_summarizer_max_items_default(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summary({1, 2, 3, 4, 5, 6, 7}, registry)
    # Check that it starts correctly and ends with ellipsis
    assert result.startswith("<class 'set'> (length=7)\n")
    assert result.endswith("\n  ...")
    # Check that there are exactly 5 items shown (default max_items)
    assert result.count("\n  (") == 5


def test_set_summarizer_max_items_custom(registry: SummarizerRegistry) -> None:
    result = SetSummarizer(max_items=2).summary({1, 2, 3, 4, 5}, registry)
    assert result.startswith("<class 'set'> (length=5)\n")
    assert result.endswith("\n  ...")
    assert result.count("\n  (") == 2


def test_set_summarizer_max_items_zero(registry: SummarizerRegistry) -> None:
    result = SetSummarizer(max_items=0).summary({1, 2, 3}, registry)
    assert result == "<class 'set'> (length=3) ..."


def test_set_summarizer_max_items_negative(registry: SummarizerRegistry) -> None:
    result = SetSummarizer(max_items=-1).summary({1, 2, 3, 4, 5, 6, 7}, registry)
    assert result.startswith("<class 'set'> (length=7)\n")
    assert not result.endswith("\n...")
    assert result.count("\n  (") == 7


def test_set_summarizer_frozenset(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summary(frozenset({1, 2}), registry)
    # Check both possible orderings
    assert result in [
        "<class 'frozenset'> (length=2)\n  (0): 1\n  (1): 2",
        "<class 'frozenset'> (length=2)\n  (0): 2\n  (1): 1",
    ]


def test_set_summarizer_nested_sets(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summary({frozenset({1, 2}), frozenset({3, 4})}, registry)
    assert result.startswith("<class 'set'> (length=2)\n")
    assert "frozenset" in result
    assert result.count("frozenset") == 2


def test_set_summarizer_depth_limit(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summary({1, 2, 3}, registry, depth=1, max_depth=1)
    # At max depth, it should return the string representation
    assert result in ["{1, 2, 3}", "{1, 3, 2}", "{2, 1, 3}", "{2, 3, 1}", "{3, 2, 1}", "{3, 1, 2}"]


def test_set_summarizer_num_spaces(registry: SummarizerRegistry) -> None:
    result = SetSummarizer(num_spaces=4).summary({1}, registry)
    assert result == "<class 'set'> (length=1)\n    (0): 1"


def test_set_summarizer_string_items(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summary({"a"}, registry)
    assert result == "<class 'set'> (length=1)\n  (0): a"


def test_set_summarizer_large_length_formatting(registry: SummarizerRegistry) -> None:
    result = SetSummarizer(max_items=2).summary(set(range(1000)), registry)
    assert result.startswith("<class 'set'> (length=1,000)\n")
    assert result.endswith("\n  ...")


def test_set_summarizer_max_items_equals_length(registry: SummarizerRegistry) -> None:
    result = SetSummarizer(max_items=3).summary({1, 2, 3}, registry)
    assert result.startswith("<class 'set'> (length=3)\n")
    assert not result.endswith("\n  ...")
    assert result.count("\n  (") == 3


def test_set_summarizer_max_items_greater_than_length(registry: SummarizerRegistry) -> None:
    result = SetSummarizer(max_items=10).summary({1, 2, 3}, registry)
    assert result.startswith("<class 'set'> (length=3)\n")
    assert not result.endswith("\n...")
    assert result.count("\n  (") == 3
