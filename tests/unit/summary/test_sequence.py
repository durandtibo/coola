from __future__ import annotations

from collections.abc import Sequence

import pytest

from coola.summary import DefaultSummarizer, SequenceSummarizer, SummarizerRegistry


@pytest.fixture
def registry() -> SummarizerRegistry:
    return SummarizerRegistry(
        {
            object: DefaultSummarizer(),
            Sequence: SequenceSummarizer(),
            list: SequenceSummarizer(),
            tuple: SequenceSummarizer(),
        }
    )


#######################################
#     Tests for SequenceSummarizer     #
#######################################


def test_sequence_summarizer_init_default() -> None:
    summarizer = SequenceSummarizer()
    assert summarizer._max_items == 5
    assert summarizer._num_spaces == 2


def test_sequence_summarizer_init_custom() -> None:
    summarizer = SequenceSummarizer(max_items=10, num_spaces=4)
    assert summarizer._max_items == 10
    assert summarizer._num_spaces == 4


def test_sequence_summarizer_init_negative_max_items() -> None:
    summarizer = SequenceSummarizer(max_items=-1)
    assert summarizer._max_items == -1


def test_sequence_summarizer_repr() -> None:
    summarizer = SequenceSummarizer()
    assert repr(summarizer) == "SequenceSummarizer(max_items=5, num_spaces=2)"


def test_sequence_summarizer_repr_large_max_items() -> None:
    summarizer = SequenceSummarizer(max_items=1000)
    assert repr(summarizer) == "SequenceSummarizer(max_items=1,000, num_spaces=2)"


def test_sequence_summarizer_str() -> None:
    summarizer = SequenceSummarizer()
    assert str(summarizer) == "SequenceSummarizer(max_items=5, num_spaces=2)"


def test_sequence_summarizer_str_large_max_items() -> None:
    summarizer = SequenceSummarizer(max_items=1000)
    assert str(summarizer) == "SequenceSummarizer(max_items=1,000, num_spaces=2)"


def test_sequence_summarizer_equal_true() -> None:
    assert SequenceSummarizer(max_items=10, num_spaces=4).equal(
        SequenceSummarizer(max_items=10, num_spaces=4)
    )


def test_sequence_summarizer_equal_true_default() -> None:
    assert SequenceSummarizer().equal(SequenceSummarizer())


def test_sequence_summarizer_equal_false_different_max_items() -> None:
    assert not SequenceSummarizer(max_items=10, num_spaces=4).equal(
        SequenceSummarizer(max_items=5, num_spaces=4)
    )


def test_sequence_summarizer_equal_false_different_num_spaces() -> None:
    assert not SequenceSummarizer(max_items=10, num_spaces=4).equal(
        SequenceSummarizer(max_items=10, num_spaces=2)
    )


def test_sequence_summarizer_equal_false_different_type() -> None:
    assert not SequenceSummarizer().equal(42)


def test_sequence_summarizer_equal_false_different_type_child() -> None:
    class Child(SequenceSummarizer): ...

    assert not SequenceSummarizer().equal(Child())


def test_sequence_summarizer_equal_false_none() -> None:
    assert not SequenceSummarizer().equal(None)


def test_sequence_summarizer_summarize_empty_list(registry: SummarizerRegistry) -> None:
    """Test summarizing an empty list."""
    summarizer = SequenceSummarizer()
    result = summarizer.summarize([], registry)
    assert result == "<class 'list'> []"


def test_sequence_summarizer_summarize_empty_tuple(registry: SummarizerRegistry) -> None:
    """Test summarizing an empty tuple."""
    summarizer = SequenceSummarizer()
    result = summarizer.summarize((), registry)
    assert result == "<class 'tuple'> ()"


def test_sequence_summarizer_summarize_simple_list(registry: SummarizerRegistry) -> None:
    """Test summarizing a simple list with few items."""
    summarizer = SequenceSummarizer()
    result = summarizer.summarize([1, 2, 3], registry)
    assert result == "<class 'list'> (length=3)\n  (0): 1\n  (1): 2\n  (2): 3"


def test_sequence_summarizer_summarize_simple_tuple(registry: SummarizerRegistry) -> None:
    """Test summarizing a simple tuple."""
    summarizer = SequenceSummarizer()
    result = summarizer.summarize((1, 2, 3), registry)
    assert result == "<class 'tuple'> (length=3)\n  (0): 1\n  (1): 2\n  (2): 3"


def test_sequence_summarizer_summarize_respects_max_items(registry: SummarizerRegistry) -> None:
    """Test that summary respects max_items limit."""
    summarizer = SequenceSummarizer(max_items=3)
    result = summarizer.summarize([1, 2, 3, 4, 5], registry)
    assert result == "<class 'list'> (length=5)\n  (0): 1\n  (1): 2\n  (2): 3\n  ..."


def test_sequence_summarizer_summarize_max_items_zero(registry: SummarizerRegistry) -> None:
    """Test summary with max_items=0 shows only type and length."""
    summarizer = SequenceSummarizer(max_items=0)
    result = summarizer.summarize([1, 2, 3], registry)
    assert result == "<class 'list'> (length=3) ..."


def test_sequence_summarizer_summarize_max_items_negative_shows_all(
    registry: SummarizerRegistry,
) -> None:
    """Test that negative max_items shows all items."""
    result = SequenceSummarizer(max_items=-1).summarize(list(range(10)), registry)
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


def test_sequence_summarizer_summarize_depth_limit_reached(registry: SummarizerRegistry) -> None:
    """Test that depth limit causes fallback to string
    representation."""
    result = SequenceSummarizer().summarize([1, 2, 3], registry, depth=1, max_depth=1)
    assert result == "[1, 2, 3]"


def test_sequence_summarizer_summarize_depth_limit_not_reached(
    registry: SummarizerRegistry,
) -> None:
    """Test normal behavior when depth limit is not reached."""
    result = SequenceSummarizer().summarize([1, 2, 3], registry, depth=0, max_depth=2)
    assert result == (
        "<class 'list'> (length=3)\n"
        "  (0): <class 'int'> 1\n"
        "  (1): <class 'int'> 2\n"
        "  (2): <class 'int'> 3"
    )


def test_sequence_summarizer_summarize_nested_structures(registry: SummarizerRegistry) -> None:
    """Test summarizing nested structures."""
    result = SequenceSummarizer().summarize([[1, 2], [3, 4]], registry, max_depth=2)
    assert result == (
        "<class 'list'> (length=2)\n"
        "  (0): <class 'list'> (length=2)\n"
        "      (0): 1\n"
        "      (1): 2\n"
        "  (1): <class 'list'> (length=2)\n"
        "      (0): 3\n"
        "      (1): 4"
    )


def test_sequence_summarizer_summarize_large_length_formatting(
    registry: SummarizerRegistry,
) -> None:
    """Test that large lengths are formatted with commas."""
    result = SequenceSummarizer(max_items=2).summarize(list(range(1000)), registry)
    assert result == "<class 'list'> (length=1,000)\n  (0): 0\n  (1): 1\n  ..."


def test_sequence_summarizer_summarize_string_sequence(registry: SummarizerRegistry) -> None:
    """Test summarizing a list of strings."""
    result = SequenceSummarizer().summarize(["a", "b", "c"], registry)
    assert result == "<class 'list'> (length=3)\n  (0): a\n  (1): b\n  (2): c"


def test_sequence_summarizer_summarize_mixed_types(registry: SummarizerRegistry) -> None:
    """Test summarizing a list with mixed types."""
    result = SequenceSummarizer().summarize([1, "two", 3.0, None], registry)
    assert result == "<class 'list'> (length=4)\n  (0): 1\n  (1): two\n  (2): 3.0\n  (3): None"


def test_sequence_summarizer_summarize_mixed_types_max_depth_2(
    registry: SummarizerRegistry,
) -> None:
    """Test summarizing a list with mixed types."""
    result = SequenceSummarizer().summarize([1, "two", 3.0, None], registry, max_depth=2)
    assert result == (
        "<class 'list'> (length=4)\n"
        "  (0): <class 'int'> 1\n"
        "  (1): <class 'str'> two\n"
        "  (2): <class 'float'> 3.0\n"
        "  (3): <class 'NoneType'> None"
    )


def test_sequence_summarizer_summarize_exactly_max_items(registry: SummarizerRegistry) -> None:
    """Test when sequence length equals max_items (no truncation)."""
    result = SequenceSummarizer(max_items=3).summarize([1, 2, 3], registry)
    assert result == "<class 'list'> (length=3)\n  (0): 1\n  (1): 2\n  (2): 3"


def test_sequence_summarizer_summarize_one_more_than_max_items(
    registry: SummarizerRegistry,
) -> None:
    """Test when sequence has one more item than max_items."""
    result = SequenceSummarizer(max_items=3).summarize([1, 2, 3, 4], registry)
    assert result == "<class 'list'> (length=4)\n  (0): 1\n  (1): 2\n  (2): 3\n  ..."


def test_sequence_summarizer_summarize_custom_num_spaces_indentation(
    registry: SummarizerRegistry,
) -> None:
    """Test that custom num_spaces affects indentation."""
    result = SequenceSummarizer(num_spaces=4).summarize([1, 2], registry)
    assert result == "<class 'list'> (length=2)\n    (0): 1\n    (1): 2"


def test_sequence_summarizer_summarize_single_item_list(registry: SummarizerRegistry) -> None:
    """Test summarizing a list with a single item."""
    result = SequenceSummarizer().summarize([42], registry)
    assert result == "<class 'list'> (length=1)\n  (0): 42"
