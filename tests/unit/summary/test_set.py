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


#######################################
#     Tests for SetSummarizer     #
#######################################


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


def test_set_summarizer_summary_empty_list(registry: SummarizerRegistry) -> None:
    """Test summarizing an empty list."""
    summarizer = SetSummarizer()
    result = summarizer.summary([], registry)
    assert result == "<class 'list'> []"


def test_set_summarizer_summary_empty_tuple(registry: SummarizerRegistry) -> None:
    """Test summarizing an empty tuple."""
    summarizer = SetSummarizer()
    result = summarizer.summary((), registry)
    assert result == "<class 'tuple'> ()"


def test_set_summarizer_summary_simple_list(registry: SummarizerRegistry) -> None:
    """Test summarizing a simple list with few items."""
    summarizer = SetSummarizer()
    result = summarizer.summary([1, 2, 3], registry)
    assert result == "<class 'list'> (length=3)\n  (0): 1\n  (1): 2\n  (2): 3"


def test_set_summarizer_summary_simple_tuple(registry: SummarizerRegistry) -> None:
    """Test summarizing a simple tuple."""
    summarizer = SetSummarizer()
    result = summarizer.summary((1, 2, 3), registry)
    assert result == "<class 'tuple'> (length=3)\n  (0): 1\n  (1): 2\n  (2): 3"


def test_set_summarizer_summary_respects_max_items(registry: SummarizerRegistry) -> None:
    """Test that summary respects max_items limit."""
    summarizer = SetSummarizer(max_items=3)
    result = summarizer.summary([1, 2, 3, 4, 5], registry)
    assert result == "<class 'list'> (length=5)\n  (0): 1\n  (1): 2\n  (2): 3\n  ..."


def test_set_summarizer_summary_max_items_zero(registry: SummarizerRegistry) -> None:
    """Test summary with max_items=0 shows only type and length."""
    summarizer = SetSummarizer(max_items=0)
    result = summarizer.summary([1, 2, 3], registry)
    assert result == "<class 'list'> (length=3) ..."


def test_set_summarizer_summary_max_items_negative_shows_all(
    registry: SummarizerRegistry,
) -> None:
    """Test that negative max_items shows all items."""
    result = SetSummarizer(max_items=-1).summary(list(range(10)), registry)
    assert result == (
        "<class 'list'> (length=10)\n"
        "  (0): 0\n"
        "  (1): 1\n"
        "  (2): 2\n"
        "  (3): 3\n"
        "  (4): 4\n"
        "  (5): 5\n"
        "  (6): 6\n"
        "  (7): 7\n"
        "  (8): 8\n"
        "  (9): 9"
    )


def test_set_summarizer_summary_depth_limit_reached(registry: SummarizerRegistry) -> None:
    """Test that depth limit causes fallback to string
    representation."""
    result = SetSummarizer().summary([1, 2, 3], registry, depth=1, max_depth=1)
    assert result == "[1, 2, 3]"


def test_set_summarizer_summary_depth_limit_not_reached(registry: SummarizerRegistry) -> None:
    """Test normal behavior when depth limit is not reached."""
    result = SetSummarizer().summary([1, 2, 3], registry, depth=0, max_depth=2)
    assert result == (
        "<class 'list'> (length=3)\n"
        "  (0): <class 'int'> 1\n"
        "  (1): <class 'int'> 2\n"
        "  (2): <class 'int'> 3"
    )


def test_set_summarizer_summary_nested_structures(registry: SummarizerRegistry) -> None:
    """Test summarizing nested structures."""
    result = SetSummarizer().summary([[1, 2], [3, 4]], registry, max_depth=2)
    assert result == (
        "<class 'list'> (length=2)\n"
        "  (0): <class 'list'> (length=2)\n"
        "      (0): 1\n"
        "      (1): 2\n"
        "  (1): <class 'list'> (length=2)\n"
        "      (0): 3\n"
        "      (1): 4"
    )


def test_set_summarizer_summary_large_length_formatting(registry: SummarizerRegistry) -> None:
    """Test that large lengths are formatted with commas."""
    result = SetSummarizer(max_items=2).summary(list(range(1000)), registry)
    assert result == "<class 'list'> (length=1,000)\n  (0): 0\n  (1): 1\n  ..."


def test_set_summarizer_summary_string_set(registry: SummarizerRegistry) -> None:
    """Test summarizing a list of strings."""
    result = SetSummarizer().summary(["a", "b", "c"], registry)
    assert result == "<class 'list'> (length=3)\n  (0): a\n  (1): b\n  (2): c"


def test_set_summarizer_summary_mixed_types(registry: SummarizerRegistry) -> None:
    """Test summarizing a list with mixed types."""
    result = SetSummarizer().summary([1, "two", 3.0, None], registry)
    assert result == "<class 'list'> (length=4)\n  (0): 1\n  (1): two\n  (2): 3.0\n  (3): None"


def test_set_summarizer_summary_mixed_types_max_depth_2(registry: SummarizerRegistry) -> None:
    """Test summarizing a list with mixed types."""
    result = SetSummarizer().summary([1, "two", 3.0, None], registry, max_depth=2)
    assert result == (
        "<class 'list'> (length=4)\n"
        "  (0): <class 'int'> 1\n"
        "  (1): <class 'str'> two\n"
        "  (2): <class 'float'> 3.0\n"
        "  (3): <class 'NoneType'> None"
    )


def test_set_summarizer_summary_exactly_max_items(registry: SummarizerRegistry) -> None:
    """Test when set length equals max_items (no truncation)."""
    result = SetSummarizer(max_items=3).summary([1, 2, 3], registry)
    assert result == "<class 'list'> (length=3)\n  (0): 1\n  (1): 2\n  (2): 3"


def test_set_summarizer_summary_one_more_than_max_items(registry: SummarizerRegistry) -> None:
    """Test when set has one more item than max_items."""
    result = SetSummarizer(max_items=3).summary([1, 2, 3, 4], registry)
    assert result == "<class 'list'> (length=4)\n  (0): 1\n  (1): 2\n  (2): 3\n  ..."


def test_set_summarizer_summary_custom_num_spaces_indentation(
    registry: SummarizerRegistry,
) -> None:
    """Test that custom num_spaces affects indentation."""
    result = SetSummarizer(num_spaces=4).summary([1, 2], registry)
    assert result == "<class 'list'> (length=2)\n    (0): 1\n    (1): 2"


def test_set_summarizer_summary_single_item_list(registry: SummarizerRegistry) -> None:
    """Test summarizing a list with a single item."""
    result = SetSummarizer().summary([42], registry)
    assert result == "<class 'list'> (length=1)\n  (0): 42"
