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


def test_set_summarizer_init_default_max_items() -> None:
    summarizer = SetSummarizer()
    assert summarizer._max_items == 5


def test_set_summarizer_init_default_num_spaces() -> None:
    summarizer = SetSummarizer()
    assert summarizer._num_spaces == 2


def test_set_summarizer_init_custom_max_items() -> None:
    summarizer = SetSummarizer(max_items=10, num_spaces=4)
    assert summarizer._max_items == 10


def test_set_summarizer_init_custom_num_spaces() -> None:
    summarizer = SetSummarizer(max_items=10, num_spaces=4)
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


def test_set_summarizer_summarize_empty_set(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summarize(set(), registry)
    assert result == "<class 'set'> set()"


def test_set_summarizer_summarize_single_item(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summarize({1}, registry)
    assert result == "<class 'set'> (length=1)\n  (0): 1"


def test_set_summarizer_summarize_multiple_items(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summarize({1, 2, 3}, registry)
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


def test_set_summarizer_summarize_max_items_default_header(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summarize({1, 2, 3, 4, 5, 6, 7}, registry)
    # Check that it starts correctly and ends with ellipsis
    assert result.startswith("<class 'set'> (length=7)\n")


def test_set_summarizer_summarize_max_items_default_ellipsis(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summarize({1, 2, 3, 4, 5, 6, 7}, registry)
    assert result.endswith("\n  ...")


def test_set_summarizer_summarize_max_items_default_shown_items(
    registry: SummarizerRegistry,
) -> None:
    result = SetSummarizer().summarize({1, 2, 3, 4, 5, 6, 7}, registry)
    # Check that there are exactly 5 items shown (default max_items)
    assert result.count("\n  (") == 5


def test_set_summarizer_summarize_max_items_custom_header(registry: SummarizerRegistry) -> None:
    result = SetSummarizer(max_items=2).summarize({1, 2, 3, 4, 5}, registry)
    assert result.startswith("<class 'set'> (length=5)\n")


def test_set_summarizer_summarize_max_items_custom_ellipsis(registry: SummarizerRegistry) -> None:
    result = SetSummarizer(max_items=2).summarize({1, 2, 3, 4, 5}, registry)
    assert result.endswith("\n  ...")


def test_set_summarizer_summarize_max_items_custom_shown_items(
    registry: SummarizerRegistry,
) -> None:
    result = SetSummarizer(max_items=2).summarize({1, 2, 3, 4, 5}, registry)
    assert result.count("\n  (") == 2


def test_set_summarizer_summarize_max_items_zero(registry: SummarizerRegistry) -> None:
    result = SetSummarizer(max_items=0).summarize({1, 2, 3}, registry)
    assert result == "<class 'set'> (length=3) ..."


def test_set_summarizer_summarize_max_items_negative_header(registry: SummarizerRegistry) -> None:
    result = SetSummarizer(max_items=-1).summarize({1, 2, 3, 4, 5, 6, 7}, registry)
    assert result.startswith("<class 'set'> (length=7)\n")


def test_set_summarizer_summarize_max_items_negative_no_ellipsis(
    registry: SummarizerRegistry,
) -> None:
    result = SetSummarizer(max_items=-1).summarize({1, 2, 3, 4, 5, 6, 7}, registry)
    assert not result.endswith("\n...")


def test_set_summarizer_summarize_max_items_negative_shown_items(
    registry: SummarizerRegistry,
) -> None:
    result = SetSummarizer(max_items=-1).summarize({1, 2, 3, 4, 5, 6, 7}, registry)
    assert result.count("\n  (") == 7


def test_set_summarizer_summarize_frozenset(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summarize(frozenset({1, 2}), registry)
    # Check both possible orderings
    assert result in [
        "<class 'frozenset'> (length=2)\n  (0): 1\n  (1): 2",
        "<class 'frozenset'> (length=2)\n  (0): 2\n  (1): 1",
    ]


def test_set_summarizer_summarize_nested_sets_header(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summarize({frozenset({1, 2}), frozenset({3, 4})}, registry)
    assert result.startswith("<class 'set'> (length=2)\n")


def test_set_summarizer_summarize_nested_sets_contains_frozensets(
    registry: SummarizerRegistry,
) -> None:
    result = SetSummarizer().summarize({frozenset({1, 2}), frozenset({3, 4})}, registry)
    assert "frozenset" in result


def test_set_summarizer_summarize_nested_sets_number_of_frozensets(
    registry: SummarizerRegistry,
) -> None:
    result = SetSummarizer().summarize({frozenset({1, 2}), frozenset({3, 4})}, registry)
    assert result.count("frozenset") == 2


def test_set_summarizer_summarize_depth_limit(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summarize({1, 2, 3}, registry, depth=1, max_depth=1)
    # At max depth, it should return the string representation
    assert result in ["{1, 2, 3}", "{1, 3, 2}", "{2, 1, 3}", "{2, 3, 1}", "{3, 2, 1}", "{3, 1, 2}"]


def test_set_summarizer_summarize_depth_limit_negative_max_items_is_capped_ellipsis(
    registry: SummarizerRegistry,
) -> None:
    """Regression test for issue #27: max_items=-1 must not dump an
    unbounded str(data) when the depth limit is reached."""
    result = SetSummarizer(max_items=-1).summarize(
        set(range(100_000)), registry, depth=1, max_depth=1
    )
    assert result.endswith(" ...")


def test_set_summarizer_summarize_depth_limit_negative_max_items_is_capped_length(
    registry: SummarizerRegistry,
) -> None:
    result = SetSummarizer(max_items=-1).summarize(
        set(range(100_000)), registry, depth=1, max_depth=1
    )
    assert len(result) < 1_000


def test_set_summarizer_summarize_num_spaces(registry: SummarizerRegistry) -> None:
    result = SetSummarizer(num_spaces=4).summarize({1}, registry)
    assert result == "<class 'set'> (length=1)\n    (0): 1"


def test_set_summarizer_summarize_string_items(registry: SummarizerRegistry) -> None:
    result = SetSummarizer().summarize({"a"}, registry)
    assert result == "<class 'set'> (length=1)\n  (0): a"


def test_set_summarizer_summarize_large_length_formatting_header(
    registry: SummarizerRegistry,
) -> None:
    result = SetSummarizer(max_items=2).summarize(set(range(1000)), registry)
    assert result.startswith("<class 'set'> (length=1,000)\n")


def test_set_summarizer_summarize_large_length_formatting_ellipsis(
    registry: SummarizerRegistry,
) -> None:
    result = SetSummarizer(max_items=2).summarize(set(range(1000)), registry)
    assert result.endswith("\n  ...")


def test_set_summarizer_summarize_depth_exceeds_max_depth_truncates_before_stringify_ellipsis(
    registry: SummarizerRegistry,
) -> None:
    """At the max_depth boundary, only the first max_items items are
    stringified, instead of stringifying the whole set."""
    data = set(range(1000))
    result = SetSummarizer(max_items=3).summarize(data, registry, depth=1, max_depth=1)
    assert result.endswith(" ...")


def test_set_summarizer_summarize_depth_exceeds_max_depth_truncates_before_stringify_shown_items(
    registry: SummarizerRegistry,
) -> None:
    data = set(range(1000))
    result = SetSummarizer(max_items=3).summarize(data, registry, depth=1, max_depth=1)
    # only 3 items were stringified, not the full 1000-item set
    assert result.count(",") == 2


def test_set_summarizer_summarize_max_items_equals_length_header(
    registry: SummarizerRegistry,
) -> None:
    result = SetSummarizer(max_items=3).summarize({1, 2, 3}, registry)
    assert result.startswith("<class 'set'> (length=3)\n")


def test_set_summarizer_summarize_max_items_equals_length_no_ellipsis(
    registry: SummarizerRegistry,
) -> None:
    result = SetSummarizer(max_items=3).summarize({1, 2, 3}, registry)
    assert not result.endswith("\n  ...")


def test_set_summarizer_summarize_max_items_equals_length_shown_items(
    registry: SummarizerRegistry,
) -> None:
    result = SetSummarizer(max_items=3).summarize({1, 2, 3}, registry)
    assert result.count("\n  (") == 3


def test_set_summarizer_summarize_max_items_greater_than_length_header(
    registry: SummarizerRegistry,
) -> None:
    result = SetSummarizer(max_items=10).summarize({1, 2, 3}, registry)
    assert result.startswith("<class 'set'> (length=3)\n")


def test_set_summarizer_summarize_max_items_greater_than_length_no_ellipsis(
    registry: SummarizerRegistry,
) -> None:
    result = SetSummarizer(max_items=10).summarize({1, 2, 3}, registry)
    assert not result.endswith("\n...")


def test_set_summarizer_summarize_max_items_greater_than_length_shown_items(
    registry: SummarizerRegistry,
) -> None:
    result = SetSummarizer(max_items=10).summarize({1, 2, 3}, registry)
    assert result.count("\n  (") == 3


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({1, 2, 3, 4}, "{1, 2} ..."),
        (frozenset({1, 2, 3, 4}), "frozenset({1, 2}) ..."),
        ({1: 0, 2: 0, 3: 0, 4: 0}.keys(), "[1, 2] ..."),
    ],
)
def test_set_summarizer_summarize_depth_limit_preview_keeps_the_type(
    registry: SummarizerRegistry, data: AbstractSet, expected: str
) -> None:
    result = SetSummarizer(max_items=2).summarize(data, registry, depth=1, max_depth=1)
    assert result == expected
