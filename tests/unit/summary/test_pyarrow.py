from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from coola.summary import (
    PyarrowArraySummarizer,
    PyarrowTableSummarizer,
    SummaryConfig,
    summarize,
)
from coola.summary.pyarrow import get_default_summarizers
from coola.testing.marks import pyarrow_available
from coola.utils.imports import MissingPackageError, is_pyarrow_available

if is_pyarrow_available():
    import pyarrow as pa
else:  # pragma: no cover
    pa = Mock()


SUMMARIZERS = [
    PyarrowArraySummarizer,
    PyarrowTableSummarizer,
]

CONFIG = SummaryConfig()
SHOW_DATA = SummaryConfig(show_data=True)


@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_missing_package_raises(summarizer_cls: type) -> None:
    with (
        patch("coola.utils.imports.optional.is_pyarrow_available", lambda: False),
        pytest.raises(MissingPackageError, match="'pyarrow' package is required"),
    ):
        summarizer_cls()


@pyarrow_available
def test_pyarrow_array_summarizer_summarize() -> None:
    result = PyarrowArraySummarizer().summarize(pa.array([1, 2, 3]), CONFIG, compact=False)
    assert result.endswith(" | length=3 | type=int64")


@pyarrow_available
def test_pyarrow_array_summarizer_summarize_chunked_array() -> None:
    result = PyarrowArraySummarizer().summarize(
        pa.chunked_array([[1, 2], [3]]), CONFIG, compact=False
    )
    assert result == "<class 'pyarrow.lib.ChunkedArray'> | length=3 | type=int64"


@pyarrow_available
def test_pyarrow_array_summarizer_summarize_show_data() -> None:
    array = pa.array([1, 2])
    assert PyarrowArraySummarizer().summarize(array, SHOW_DATA, compact=False) == repr(array)


@pyarrow_available
def test_pyarrow_table_summarizer_summarize() -> None:
    table = pa.table({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    result = PyarrowTableSummarizer().summarize(table, CONFIG, compact=False)
    assert result == "<class 'pyarrow.lib.Table'> | shape=(3, 2) | columns=a: int64, b: double"


@pyarrow_available
def test_pyarrow_table_summarizer_summarize_many_columns() -> None:
    table = pa.table({f"c{i}": [i] for i in range(13)})
    result = PyarrowTableSummarizer().summarize(table, CONFIG, compact=False)
    assert result.endswith("c4: int64, ... (+8 more)")


@pyarrow_available
def test_pyarrow_table_summarizer_summarize_show_data() -> None:
    table = pa.table({"a": [1, 2]})
    assert PyarrowTableSummarizer().summarize(table, SHOW_DATA, compact=False) == repr(table)


@pyarrow_available
def test_pyarrow_table_summarizer_summarize_record_batch() -> None:
    batch = pa.record_batch({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    result = PyarrowTableSummarizer().summarize(batch, CONFIG, compact=False)
    assert (
        result == "<class 'pyarrow.lib.RecordBatch'> | shape=(3, 2) | columns=a: int64, b: double"
    )


@pyarrow_available
def test_get_default_summarizers_record_batch_uses_the_table_summarizer() -> None:
    assert isinstance(get_default_summarizers()[pa.RecordBatch], PyarrowTableSummarizer)


@pyarrow_available
def test_summarize_record_batch_with_the_default_registry() -> None:
    result = summarize(pa.record_batch({"a": [1, 2]}))
    assert result == "<class 'pyarrow.lib.RecordBatch'> | shape=(2, 1) | columns=a: int64"
