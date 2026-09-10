from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from coola.summary import DefaultSummarizer, SummarizerRegistry
from coola.summary.collection import BaseCollectionSummarizer


class FakeSequenceSummarizer(BaseCollectionSummarizer[Sequence[Any]]):
    r"""Minimal concrete summarizer used to test the shared template
    method implemented in ``BaseCollectionSummarizer``."""

    def _get_preview(self, data: Sequence[Any]) -> list:
        return list(data[: self._max_items])

    def _format_items(
        self,
        data: Sequence[Any],
        registry: SummarizerRegistry,
        depth: int,
        max_depth: int,
    ) -> str:
        items = data if self._max_items < 0 else data[: self._max_items]
        return "\n".join(
            f"  ({i}): {registry.summarize(value, depth=depth + 1, max_depth=max_depth)}"
            for i, value in enumerate(items)
        )


@pytest.fixture
def registry() -> SummarizerRegistry:
    return SummarizerRegistry(
        {
            object: DefaultSummarizer(),
            Sequence: FakeSequenceSummarizer(),
            list: FakeSequenceSummarizer(),
        }
    )


##################################################
#     Tests for BaseCollectionSummarizer         #
##################################################


def test_base_collection_summarizer_not_implemented() -> None:
    summarizer = BaseCollectionSummarizer()
    with pytest.raises(NotImplementedError):
        summarizer._get_preview([1, 2, 3])
    with pytest.raises(NotImplementedError):
        summarizer._format_items([1, 2, 3], SummarizerRegistry(), depth=0, max_depth=1)


def test_base_collection_summarizer_summarize_empty(registry: SummarizerRegistry) -> None:
    summarizer = FakeSequenceSummarizer()
    out = summarizer.summarize([], registry)
    assert out == "<class 'list'> []"


def test_base_collection_summarizer_summarize_max_items_zero(
    registry: SummarizerRegistry,
) -> None:
    summarizer = FakeSequenceSummarizer(max_items=0)
    out = summarizer.summarize([1, 2, 3], registry)
    assert out == "<class 'list'> (length=3) ..."


def test_base_collection_summarizer_summarize_no_truncation(
    registry: SummarizerRegistry,
) -> None:
    summarizer = FakeSequenceSummarizer(max_items=5)
    out = summarizer.summarize([1, 2, 3], registry)
    assert out == "<class 'list'> (length=3)\n    (0): 1\n    (1): 2\n    (2): 3"


def test_base_collection_summarizer_summarize_truncation(registry: SummarizerRegistry) -> None:
    summarizer = FakeSequenceSummarizer(max_items=2)
    out = summarizer.summarize([1, 2, 3], registry)
    assert out == "<class 'list'> (length=3)\n    (0): 1\n    (1): 2\n  ..."


def test_base_collection_summarizer_summarize_depth_limit_no_truncation(
    registry: SummarizerRegistry,
) -> None:
    summarizer = FakeSequenceSummarizer(max_items=5)
    out = summarizer.summarize([1, 2, 3], registry, depth=1, max_depth=1)
    assert out == "[1, 2, 3]"


def test_base_collection_summarizer_summarize_depth_limit_with_preview(
    registry: SummarizerRegistry,
) -> None:
    summarizer = FakeSequenceSummarizer(max_items=2)
    out = summarizer.summarize([1, 2, 3], registry, depth=1, max_depth=1)
    assert out == "[1, 2] ..."
