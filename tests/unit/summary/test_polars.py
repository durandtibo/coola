from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from coola.summary import (
    PolarsDataFrameSummarizer,
    PolarsLazyFrameSummarizer,
    PolarsSeriesSummarizer,
    SummaryConfig,
)
from coola.testing.marks import polars_available
from coola.utils.imports import MissingPackageError, is_polars_available

if is_polars_available():
    import polars as pl
else:  # pragma: no cover
    pl = Mock()


SUMMARIZERS = [
    PolarsDataFrameSummarizer,
    PolarsLazyFrameSummarizer,
    PolarsSeriesSummarizer,
]

CONFIG = SummaryConfig()
SHOW_DATA = SummaryConfig(show_data=True)


@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_missing_package_raises(summarizer_cls: type) -> None:
    with (
        patch("coola.utils.imports.optional.is_polars_available", lambda: False),
        pytest.raises(MissingPackageError, match="'polars' package is required"),
    ):
        summarizer_cls()


@polars_available
def test_polars_dataframe_summarizer_summarize_columns() -> None:
    frame = pl.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    result = PolarsDataFrameSummarizer().summarize(frame, CONFIG, compact=False)
    assert result.endswith(" | shape=(3, 2) | columns=a: Int64, b: Float64")


@polars_available
def test_polars_dataframe_summarizer_summarize_class_name() -> None:
    frame = pl.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    result = PolarsDataFrameSummarizer().summarize(frame, CONFIG, compact=False)
    assert "DataFrame" in result.split(" | ")[0]


@polars_available
def test_polars_dataframe_summarizer_summarize_many_columns() -> None:
    frame = pl.DataFrame({f"c{i}": [i] for i in range(13)})
    result = PolarsDataFrameSummarizer().summarize(frame, CONFIG, compact=False)
    assert result.endswith("c4: Int64, ... (+8 more)")


@polars_available
def test_polars_dataframe_summarizer_summarize_show_data() -> None:
    frame = pl.DataFrame({"a": [1, 2]})
    assert PolarsDataFrameSummarizer().summarize(frame, SHOW_DATA, compact=False) == repr(frame)


@polars_available
def test_polars_lazyframe_summarizer_summarize_does_not_collect() -> None:
    frame = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    with patch.object(pl.LazyFrame, "collect", side_effect=AssertionError("collected")):
        result = PolarsLazyFrameSummarizer().summarize(frame, CONFIG, compact=False)
    assert result.endswith(" | columns=a: Int64, b: String")


@polars_available
def test_polars_lazyframe_summarizer_summarize_class_name() -> None:
    frame = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = PolarsLazyFrameSummarizer().summarize(frame, CONFIG, compact=False)
    assert "LazyFrame" in result.split(" | ")[0]


@polars_available
def test_polars_lazyframe_summarizer_summarize_invalid_plan() -> None:
    frame = pl.LazyFrame({"a": [1, 2, 3]}).select("missing")
    result = PolarsLazyFrameSummarizer().summarize(frame, CONFIG, compact=False)
    assert result.endswith(" | columns=<unresolved schema: ColumnNotFoundError>")


@polars_available
def test_polars_lazyframe_summarizer_summarize_show_data() -> None:
    frame = pl.LazyFrame({"a": [1, 2]})
    assert PolarsLazyFrameSummarizer().summarize(frame, SHOW_DATA, compact=False) == repr(frame)


@polars_available
def test_polars_series_summarizer_summarize_columns() -> None:
    result = PolarsSeriesSummarizer().summarize(pl.Series("a", [1, 2, 3]), CONFIG, compact=False)
    assert result.endswith(" | shape=(3,) | dtype=Int64 | name='a'")


@polars_available
def test_polars_series_summarizer_summarize_class_name() -> None:
    result = PolarsSeriesSummarizer().summarize(pl.Series("a", [1, 2, 3]), CONFIG, compact=False)
    assert "Series" in result.split(" | ")[0]


@polars_available
def test_polars_series_summarizer_summarize_show_data() -> None:
    series = pl.Series("a", [1, 2])
    assert PolarsSeriesSummarizer().summarize(series, SHOW_DATA, compact=False) == repr(series)
