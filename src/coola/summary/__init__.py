r"""Utilities to build readable summaries for nested data structures."""

from __future__ import annotations

__all__ = [
    "BaseCollectionSummarizer",
    "BaseSummarizer",
    "DefaultSummarizer",
    "JaxArraySummarizer",
    "MappingSummarizer",
    "NDArraySummarizer",
    "PandasDataFrameSummarizer",
    "PandasSeriesSummarizer",
    "PolarsDataFrameSummarizer",
    "PolarsLazyFrameSummarizer",
    "PolarsSeriesSummarizer",
    "PyarrowArraySummarizer",
    "PyarrowTableSummarizer",
    "SequenceSummarizer",
    "SetSummarizer",
    "Summarizer",
    "SummarizerRegistry",
    "TensorSummarizer",
    "XarrayDataArraySummarizer",
    "XarrayDatasetSummarizer",
    "XarrayVariableSummarizer",
    "get_default_registry",
    "get_default_summarizer_registry",
    "register_summarizers",
    "summarize",
]

from coola.summary.base import BaseSummarizer, Summarizer
from coola.summary.collection import BaseCollectionSummarizer
from coola.summary.default import DefaultSummarizer
from coola.summary.interface import (
    get_default_registry,
    get_default_summarizer_registry,
    register_summarizers,
    summarize,
)
from coola.summary.jax import JaxArraySummarizer
from coola.summary.mapping import MappingSummarizer
from coola.summary.numpy import NDArraySummarizer
from coola.summary.pandas import PandasDataFrameSummarizer, PandasSeriesSummarizer
from coola.summary.polars import (
    PolarsDataFrameSummarizer,
    PolarsLazyFrameSummarizer,
    PolarsSeriesSummarizer,
)
from coola.summary.pyarrow import PyarrowArraySummarizer, PyarrowTableSummarizer
from coola.summary.registry import SummarizerRegistry
from coola.summary.sequence import SequenceSummarizer
from coola.summary.set import SetSummarizer
from coola.summary.torch import TensorSummarizer
from coola.summary.xarray import (
    XarrayDataArraySummarizer,
    XarrayDatasetSummarizer,
    XarrayVariableSummarizer,
)
