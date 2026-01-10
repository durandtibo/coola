r"""Contain functionalities to compute a text summary of nested data
based on the type of data."""

from __future__ import annotations

__all__ = [
    "BaseCollectionSummarizer",
    "BaseSummarizer",
    "DefaultSummarizer",
    "MappingSummarizer",
    "SummarizerRegistry",
]

from coola.summary.base import BaseSummarizer
from coola.summary.collection import BaseCollectionSummarizer
from coola.summary.default import DefaultSummarizer
from coola.summary.mapping import MappingSummarizer
from coola.summary.registry import SummarizerRegistry
