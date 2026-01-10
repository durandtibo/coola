from __future__ import annotations

import pytest

from coola.summary import DefaultSummarizer, SummarizerRegistry


@pytest.fixture
def registry() -> SummarizerRegistry:
    return SummarizerRegistry()


#######################################
#     Tests for DefaultSummarizer     #
#######################################


def test_default_summarizer_init_default() -> None:
    assert DefaultSummarizer()._max_characters == -1


def test_default_summarizer_init_with_max_characters() -> None:
    assert DefaultSummarizer(max_characters=50)._max_characters == 50


def test_default_summarizer_init_with_zero_max_characters() -> None:
    assert DefaultSummarizer(max_characters=0)._max_characters == 0


def test_default_summarizer_repr_default() -> None:
    assert repr(DefaultSummarizer()) == "DefaultSummarizer(max_characters=-1)"


def test_default_summarizer_repr_with_max_characters() -> None:
    assert (
        repr(DefaultSummarizer(max_characters=1_000_000))
        == "DefaultSummarizer(max_characters=1,000,000)"
    )


def test_default_summarizer_str_default() -> None:
    assert repr(DefaultSummarizer()) == "DefaultSummarizer(max_characters=-1)"


def test_default_summarizer_str_with_max_characters() -> None:
    assert (
        repr(DefaultSummarizer(max_characters=1_000_000))
        == "DefaultSummarizer(max_characters=1,000,000)"
    )


def test_default_summarizer_equal_true_same_instance() -> None:
    summarizer = DefaultSummarizer()
    assert summarizer.equal(summarizer)


def test_default_summarizer_equal_true_same_config() -> None:
    summarizer1 = DefaultSummarizer(max_characters=50)
    summarizer2 = DefaultSummarizer(max_characters=50)
    assert summarizer1.equal(summarizer2)


def test_default_summarizer_equal_false_different_max_characters() -> None:
    summarizer1 = DefaultSummarizer(max_characters=50)
    summarizer2 = DefaultSummarizer(max_characters=100)
    assert not summarizer1.equal(summarizer2)


def test_default_summarizer_equal_false_different_type() -> None:
    assert not DefaultSummarizer().equal("not a summarizer")


def test_default_summarizer_equal_false_different_type_child() -> None:
    class Child(DefaultSummarizer): ...

    assert not DefaultSummarizer().equal(Child())


def test_default_summarizer_equal_none() -> None:
    assert not DefaultSummarizer().equal(None)


def test_default_summarizer_summarize_integer(registry: SummarizerRegistry) -> None:
    summarizer = DefaultSummarizer()
    result = summarizer.summarize(1, registry)
    assert result == "<class 'int'> 1"


def test_default_summarizer_summarize_string(registry: SummarizerRegistry) -> None:
    summarizer = DefaultSummarizer()
    result = summarizer.summarize("hello", registry)
    assert result == "<class 'str'> hello"


def test_default_summarizer_summarize_list(registry: SummarizerRegistry) -> None:
    summarizer = DefaultSummarizer()
    result = summarizer.summarize([1, 2, 3], registry)
    assert result == "<class 'list'> [1, 2, 3]"


def test_default_summarizer_summarize_depth_zero(registry: SummarizerRegistry) -> None:
    summarizer = DefaultSummarizer()
    result = summarizer.summarize("test", registry, depth=0, max_depth=1)
    assert result == "<class 'str'> test"


def test_default_summarizer_summarize_depth_at_max(registry: SummarizerRegistry) -> None:
    summarizer = DefaultSummarizer()
    result = summarizer.summarize("test", registry, depth=1, max_depth=1)
    assert result == "test"


def test_default_summarizer_summarize_depth_exceeds_max(registry: SummarizerRegistry) -> None:
    summarizer = DefaultSummarizer()
    result = summarizer.summarize("test", registry, depth=2, max_depth=1)
    assert result == "test"


def test_default_summarizer_summarize_with_max_characters_no_truncation(
    registry: SummarizerRegistry,
) -> None:
    summarizer = DefaultSummarizer(max_characters=20)
    result = summarizer.summarize("short", registry)
    assert result == "<class 'str'> short"


def test_default_summarizer_summarize_with_max_characters_truncation(
    registry: SummarizerRegistry,
) -> None:
    summarizer = DefaultSummarizer(max_characters=10)
    result = summarizer.summarize("this is a very long string", registry)
    assert result == "<class 'str'> this is a ..."


def test_default_summarizer_summarize_with_max_characters_exact_length(
    registry: SummarizerRegistry,
) -> None:
    summarizer = DefaultSummarizer(max_characters=5)
    result = summarizer.summarize("hello", registry)
    assert result == "<class 'str'> hello"


def test_default_summarizer_summarize_with_max_characters_one_over(
    registry: SummarizerRegistry,
) -> None:
    summarizer = DefaultSummarizer(max_characters=5)
    result = summarizer.summarize("hello!", registry)
    assert result == "<class 'str'> hello..."


def test_default_summarizer_summarize_none(registry: SummarizerRegistry) -> None:
    summarizer = DefaultSummarizer()
    result = summarizer.summarize(None, registry)
    assert result == "<class 'NoneType'> None"


def test_default_summarizer_summarize_custom_object(registry: SummarizerRegistry) -> None:
    class CustomObj:
        def __str__(self) -> str:
            return "custom_representation"

    summarizer = DefaultSummarizer()
    result = summarizer.summarize(CustomObj(), registry)
    assert result.endswith(".CustomObj'> custom_representation")


def test_default_summarizer_summarize_with_negative_max_characters(
    registry: SummarizerRegistry,
) -> None:
    summarizer = DefaultSummarizer(max_characters=-100)
    assert summarizer.summarize("test string", registry) == "<class 'str'> test string"
