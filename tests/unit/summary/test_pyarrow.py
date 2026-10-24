from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from coola.summary import (
    PyarrowArraySummarizer,
    PyarrowTableSummarizer,
    SummarizerRegistry,
    summarize,
)
from coola.summary.pyarrow import get_default_summarizers
from coola.testing.fixtures import pyarrow_available
from coola.utils.imports import MissingPackageError, is_pyarrow_available

if is_pyarrow_available():
    import pyarrow as pa
else:  # pragma: no cover
    pa = Mock()


@pytest.fixture
def registry() -> SummarizerRegistry:
    return SummarizerRegistry()


SUMMARIZERS = [
    PyarrowArraySummarizer,
    PyarrowTableSummarizer,
]


@pyarrow_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_init_default(summarizer_cls: type) -> None:
    assert summarizer_cls()._show_data is False


@pyarrow_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_init_show_data(summarizer_cls: type) -> None:
    assert summarizer_cls(show_data=True)._show_data is True


@pyarrow_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_repr_default(summarizer_cls: type) -> None:
    assert repr(summarizer_cls()) == f"{summarizer_cls.__name__}(show_data=False)"


@pyarrow_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_repr_show_data(summarizer_cls: type) -> None:
    assert repr(summarizer_cls(show_data=True)) == f"{summarizer_cls.__name__}(show_data=True)"


@pyarrow_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_str(summarizer_cls: type) -> None:
    assert str(summarizer_cls()) == f"{summarizer_cls.__name__}(show_data=False)"


@pyarrow_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_true_default(summarizer_cls: type) -> None:
    assert summarizer_cls().equal(summarizer_cls())


@pyarrow_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_true_show_data(summarizer_cls: type) -> None:
    assert summarizer_cls(show_data=True).equal(summarizer_cls(show_data=True))


@pyarrow_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_false_different_show_data(summarizer_cls: type) -> None:
    assert not summarizer_cls(show_data=False).equal(summarizer_cls(show_data=True))


@pyarrow_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_false_different_type(summarizer_cls: type) -> None:
    assert not summarizer_cls().equal(42)


@pyarrow_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_false_child_class(summarizer_cls: type) -> None:
    class Child(summarizer_cls): ...

    assert not summarizer_cls().equal(Child())


@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_missing_package_raises(summarizer_cls: type) -> None:
    with (
        patch("coola.utils.imports.pyarrow.is_pyarrow_available", lambda: False),
        pytest.raises(MissingPackageError, match="'pyarrow' package is required"),
    ):
        summarizer_cls()


@pyarrow_available
def test_pyarrow_array_summarizer_summarize(registry: SummarizerRegistry) -> None:
    result = PyarrowArraySummarizer().summarize(pa.array([1, 2, 3]), registry)
    assert result.endswith(" | length=3 | type=int64")


@pyarrow_available
def test_pyarrow_array_summarizer_summarize_chunked_array(registry: SummarizerRegistry) -> None:
    result = PyarrowArraySummarizer().summarize(pa.chunked_array([[1, 2], [3]]), registry)
    assert result == "<class 'pyarrow.lib.ChunkedArray'> | length=3 | type=int64"


@pyarrow_available
def test_pyarrow_array_summarizer_summarize_show_data(registry: SummarizerRegistry) -> None:
    array = pa.array([1, 2])
    assert PyarrowArraySummarizer(show_data=True).summarize(array, registry) == repr(array)


@pyarrow_available
def test_pyarrow_table_summarizer_summarize(registry: SummarizerRegistry) -> None:
    table = pa.table({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    result = PyarrowTableSummarizer().summarize(table, registry)
    assert result == "<class 'pyarrow.lib.Table'> | shape=(3, 2) | columns=a: int64, b: double"


@pyarrow_available
def test_pyarrow_table_summarizer_summarize_many_columns(registry: SummarizerRegistry) -> None:
    table = pa.table({f"c{i}": [i] for i in range(13)})
    result = PyarrowTableSummarizer().summarize(table, registry)
    assert result.endswith("c9: int64, ... (+3 more)")


@pyarrow_available
def test_pyarrow_table_summarizer_summarize_show_data(registry: SummarizerRegistry) -> None:
    table = pa.table({"a": [1, 2]})
    assert PyarrowTableSummarizer(show_data=True).summarize(table, registry) == repr(table)


@pyarrow_available
def test_table_summarizer_max_fields(registry: SummarizerRegistry) -> None:
    table = pa.table({"a": [1], "b": [2.0], "c": ["x"]})
    out = PyarrowTableSummarizer(max_fields=1).summarize(table, registry)
    assert out.endswith("columns=a: int64, ... (+2 more)")


@pyarrow_available
def test_table_summarizer_max_fields_repr() -> None:
    assert repr(PyarrowTableSummarizer(max_fields=2)) == (
        "PyarrowTableSummarizer(show_data=False, max_fields=2)"
    )


@pyarrow_available
def test_pyarrow_table_summarizer_summarize_record_batch(registry: SummarizerRegistry) -> None:
    batch = pa.record_batch({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    result = PyarrowTableSummarizer().summarize(batch, registry)
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
