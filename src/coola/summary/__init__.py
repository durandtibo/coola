r"""Utilities to build readable summaries for nested data structures.

A summary is a short text that a person reads in a log, a notebook or an error message, to know
what a (possibly huge, nested, partly unprintable) object contains.

Properties:

- Bounded output: a node shows at most ``max_items`` children, a text is cut at
    ``max_characters``, and the expansion stops at ``max_depth``. What is not shown is announced.
- Bounded work: the cost depends on the size of the output, not on the size of the data.
- Display depth: ``max_depth`` is the number of levels that are expanded. Deeper nodes are shown
    on one line. It never raises.
- Total: summarizing never fails because of the data (a ``__len__`` or a property that raises is
    replaced by a placeholder).
- Cycle-safe: a container that contains itself is shown as ``<cycle>``.
- Any depth: there is no ``RecursionError``, the engine uses an explicit stack.
- Type dispatch: the summarizer is chosen from the type of the data along the Method Resolution
    Order, so the most specific registered type wins.
- Registry-extensible: a new type is supported by registering one summarizer.
- Metadata first: an array, a tensor, a data frame shows its metadata, not its values.
- Iterators are not consumed: a generator or an iterator is a leaf.
- Deterministic: the same value gives the same text across calls and processes (a set shows its
    smallest items in order).
- Pure: the data is never modified.
- For humans: the output is meant to be read, not parsed, so its exact format can change.

Not guaranteed: a container that is reachable by several paths is shown once per path, and the
time of a user ``__str__`` or ``__repr__`` is not bounded.
"""

from __future__ import annotations

__all__ = [
    "BaseMetadataSummarizer",
    "BaseSummarizer",
    "DataclassSummarizer",
    "DefaultSummarizer",
    "JaxArraySummarizer",
    "MappingSummarizer",
    "NDArraySummarizer",
    "Node",
    "PandasDataFrameSummarizer",
    "PandasSeriesSummarizer",
    "PolarsDataFrameSummarizer",
    "PolarsLazyFrameSummarizer",
    "PolarsSeriesSummarizer",
    "PyarrowArraySummarizer",
    "PyarrowTableSummarizer",
    "PydanticModelSummarizer",
    "SequenceSummarizer",
    "SetSummarizer",
    "Summarizer",
    "SummarizerRegistry",
    "SummaryConfig",
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

from coola.summary.base import BaseSummarizer, Node, Summarizer
from coola.summary.config import SummaryConfig
from coola.summary.dataclass import DataclassSummarizer
from coola.summary.default import DefaultSummarizer
from coola.summary.interface import (
    get_default_registry,
    get_default_summarizer_registry,
    register_summarizers,
    summarize,
)
from coola.summary.mapping import MappingSummarizer
from coola.summary.metadata import BaseMetadataSummarizer, format_fields
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
    from coola.summary.pydantic import PydanticModelSummarizer
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
    "PydanticModelSummarizer": "coola.summary.pydantic",
    "PyarrowArraySummarizer": "coola.summary.pyarrow",
    "PyarrowTableSummarizer": "coola.summary.pyarrow",
    "TensorSummarizer": "coola.summary.torch",
    "XarrayDataArraySummarizer": "coola.summary.xarray",
    "XarrayDatasetSummarizer": "coola.summary.xarray",
    "XarrayVariableSummarizer": "coola.summary.xarray",
}
__getattr__, __dir__ = lazy_exports(__name__, _LAZY_EXPORTS, globals())
