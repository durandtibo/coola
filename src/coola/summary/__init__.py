r"""Utilities to build readable summaries for nested data structures."""

from __future__ import annotations

__all__ = [
    "MAX_FIELDS",
    "BaseCollectionSummarizer",
    "BaseFieldsSummarizer",
    "BaseMetadataSummarizer",
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
    "format_fields",
    "get_default_registry",
    "get_default_summarizer_registry",
    "register_summarizers",
    "summarize",
]

from typing import TYPE_CHECKING

from coola.summary.base import BaseSummarizer, Summarizer
from coola.summary.collection import BaseCollectionSummarizer
from coola.summary.default import DefaultSummarizer
from coola.summary.interface import (
    get_default_registry,
    get_default_summarizer_registry,
    register_summarizers,
    summarize,
)
from coola.summary.mapping import MappingSummarizer
from coola.summary.metadata import (
    MAX_FIELDS,
    BaseFieldsSummarizer,
    BaseMetadataSummarizer,
    format_fields,
)
from coola.summary.registry import SummarizerRegistry
from coola.summary.sequence import SequenceSummarizer
from coola.summary.set import SetSummarizer
from coola.utils.imports import lazy_exports

if TYPE_CHECKING:
    from coola.summary.jax import JaxArraySummarizer
    from coola.summary.numpy import NDArraySummarizer
    from coola.summary.pandas import PandasDataFrameSummarizer, PandasSeriesSummarizer
    from coola.summary.polars import (
        PolarsDataFrameSummarizer,
        PolarsLazyFrameSummarizer,
        PolarsSeriesSummarizer,
    )
    from coola.summary.pyarrow import PyarrowArraySummarizer, PyarrowTableSummarizer
    from coola.summary.torch import TensorSummarizer
    from coola.summary.xarray import (
        XarrayDataArraySummarizer,
        XarrayDatasetSummarizer,
        XarrayVariableSummarizer,
    )


# The names defined in the modules that import an optional library are imported the first time they
# are used, so the library is not imported with the package.
_LAZY_EXPORTS = {
    "JaxArraySummarizer": "coola.summary.jax",
    "NDArraySummarizer": "coola.summary.numpy",
    "PandasDataFrameSummarizer": "coola.summary.pandas",
    "PandasSeriesSummarizer": "coola.summary.pandas",
    "PolarsDataFrameSummarizer": "coola.summary.polars",
    "PolarsLazyFrameSummarizer": "coola.summary.polars",
    "PolarsSeriesSummarizer": "coola.summary.polars",
    "PyarrowArraySummarizer": "coola.summary.pyarrow",
    "PyarrowTableSummarizer": "coola.summary.pyarrow",
    "TensorSummarizer": "coola.summary.torch",
    "XarrayDataArraySummarizer": "coola.summary.xarray",
    "XarrayDatasetSummarizer": "coola.summary.xarray",
    "XarrayVariableSummarizer": "coola.summary.xarray",
}
__getattr__, __dir__ = lazy_exports(__name__, _LAZY_EXPORTS, globals())
