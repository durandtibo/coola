r"""Contain functionalities to compute a text summary of nested data based on the type of data."""


from __future__ import annotations

__all__ = [
    "BaseCollectionSummarizer",
    "BaseSummarizer",
    "DefaultSummarizer",
    "MappingSummarizer",
    "NDArraySummarizer",
    "SequenceSummarizer",
    "SetSummarizer",
    "SummarizerRegistry",
    "TensorSummarizer",
    "get_default_registry",
    "register_summarizers",
    "summarize",
]

from coola.summary.base import BaseSummarizer
from coola.summary.collection import BaseCollectionSummarizer
from coola.summary.default import DefaultSummarizer
from coola.summary.interface import (
    get_default_registry,
    register_summarizers,
    summarize,
)
from coola.summary.mapping import MappingSummarizer
from coola.summary.numpy import NDArraySummarizer
from coola.summary.registry import SummarizerRegistry
from coola.summary.sequence import SequenceSummarizer
from coola.summary.set import SetSummarizer
from coola.summary.torch import TensorSummarizer
