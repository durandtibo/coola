r"""Contain functionalities to compute a text summary of nested data
based on the type of data."""

from __future__ import annotations

__all__ = ["BaseSummarizer", "DefaultSummarizer", "SummarizerRegistry"]

from coola.summary.base import BaseSummarizer
from coola.summary.default import DefaultSummarizer
from coola.summary.registry import SummarizerRegistry
