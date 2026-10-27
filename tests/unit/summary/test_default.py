from __future__ import annotations

import pytest

from coola.summary import DefaultSummarizer, SummaryConfig

CONFIG = SummaryConfig()


class BadStr:
    def __str__(self) -> str:
        msg = "broken"
        raise RuntimeError(msg)


#######################################
#     Tests for DefaultSummarizer     #
#######################################


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (1, "<class 'int'> 1"),
        (1.5, "<class 'float'> 1.5"),
        (True, "<class 'bool'> True"),
        (None, "<class 'NoneType'> None"),
        ("abc", "<class 'str'> abc"),
        ("", "<class 'str'> "),
    ],
)
def test_default_summarizer_summarize(data: object, expected: str) -> None:
    assert DefaultSummarizer().summarize(data, CONFIG, compact=False) == expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [(1, "1"), ("abc", "abc"), (None, "None"), ("", "")],
)
def test_default_summarizer_summarize_compact(data: object, expected: str) -> None:
    assert DefaultSummarizer().summarize(data, CONFIG, compact=True) == expected


def test_default_summarizer_summarize_does_not_cut_the_text() -> None:
    assert DefaultSummarizer().summarize(
        "abcdef", SummaryConfig(max_characters=3), compact=True
    ) == ("abcdef")


def test_default_summarizer_summarize_multiline_text() -> None:
    assert DefaultSummarizer().summarize("a\nb", CONFIG, compact=False) == "<class 'str'> a\nb"


def test_default_summarizer_summarize_failing_str() -> None:
    result = DefaultSummarizer().summarize(BadStr(), CONFIG, compact=True)
    assert result.startswith("<BadStr object at 0x")


def test_default_summarizer_summarize_is_deterministic() -> None:
    summarizer = DefaultSummarizer()
    assert summarizer.summarize(1, CONFIG, compact=False) == summarizer.summarize(
        1, CONFIG, compact=False
    )


def test_default_summarizer_equal_same_type() -> None:
    assert DefaultSummarizer().equal(DefaultSummarizer())


def test_default_summarizer_equal_different_type() -> None:
    assert not DefaultSummarizer().equal(42)


def test_default_summarizer_repr() -> None:
    assert repr(DefaultSummarizer()) == "DefaultSummarizer()"
