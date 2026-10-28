from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from coola.summary import (
    PandasDataFrameSummarizer,
    PandasSeriesSummarizer,
    SummaryConfig,
)
from coola.testing.marks import pandas_available
from coola.utils.imports import MissingPackageError, is_pandas_available

if is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    pd = Mock()


SUMMARIZERS = [
    PandasDataFrameSummarizer,
    PandasSeriesSummarizer,
]

CONFIG = SummaryConfig()
SHOW_DATA = SummaryConfig(show_data=True)


@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_missing_package_raises(summarizer_cls: type) -> None:
    with (
        patch("coola.utils.imports.optional.is_pandas_available", lambda: False),
        pytest.raises(MissingPackageError, match="'pandas' package is required"),
    ):
        summarizer_cls()


@pandas_available
def test_pandas_dataframe_summarizer_summarize() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": ["x", "y", "z"]})
    result = PandasDataFrameSummarizer().summarize(frame, CONFIG, compact=False)
    assert result.startswith(f"{type(frame)} | shape=(3, 3) | columns=a: int64, b: float64, c: ")


@pandas_available
def test_pandas_dataframe_summarizer_summarize_empty() -> None:
    frame = pd.DataFrame()
    result = PandasDataFrameSummarizer().summarize(frame, CONFIG, compact=False)
    assert result == f"{type(frame)} | shape=(0, 0) | columns="


@pandas_available
def test_pandas_dataframe_summarizer_summarize_class_name_class_prefix() -> None:
    result = PandasDataFrameSummarizer().summarize(pd.DataFrame({"a": [1]}), CONFIG, compact=False)
    assert "<class 'pandas." in result


@pandas_available
def test_pandas_dataframe_summarizer_summarize_class_name_class_name() -> None:
    result = PandasDataFrameSummarizer().summarize(pd.DataFrame({"a": [1]}), CONFIG, compact=False)
    assert ".DataFrame'>" in result


@pandas_available
def test_pandas_dataframe_summarizer_summarize_many_columns_shows_the_first_columns() -> None:
    frame = pd.DataFrame({f"c{i}": [i] for i in range(13)})
    result = PandasDataFrameSummarizer().summarize(frame, CONFIG, compact=False)
    assert "c4: int64, ... (+8 more)" in result


@pandas_available
def test_pandas_dataframe_summarizer_summarize_many_columns_hides_the_other_columns() -> None:
    frame = pd.DataFrame({f"c{i}": [i] for i in range(13)})
    result = PandasDataFrameSummarizer().summarize(frame, CONFIG, compact=False)
    assert "c5" not in result


@pandas_available
def test_pandas_dataframe_summarizer_summarize_show_data() -> None:
    frame = pd.DataFrame({"a": [1, 2]})
    assert PandasDataFrameSummarizer().summarize(frame, SHOW_DATA, compact=False) == repr(frame)


@pandas_available
def test_pandas_series_summarizer_summarize() -> None:
    series = pd.Series([1, 2, 3], name="a")
    result = PandasSeriesSummarizer().summarize(series, CONFIG, compact=False)
    assert result == f"{type(series)} | shape=(3,) | dtype=int64 | name='a'"


@pandas_available
def test_pandas_series_summarizer_summarize_class_name_class_prefix() -> None:
    result = PandasSeriesSummarizer().summarize(pd.Series([1]), CONFIG, compact=False)
    assert "<class 'pandas." in result


@pandas_available
def test_pandas_series_summarizer_summarize_class_name_class_name() -> None:
    result = PandasSeriesSummarizer().summarize(pd.Series([1]), CONFIG, compact=False)
    assert ".Series'>" in result


@pandas_available
def test_pandas_series_summarizer_summarize_no_name() -> None:
    series = pd.Series([1.0, 2.0])
    result = PandasSeriesSummarizer().summarize(series, CONFIG, compact=False)
    assert result == f"{type(series)} | shape=(2,) | dtype=float64 | name=None"


@pandas_available
def test_pandas_series_summarizer_summarize_show_data() -> None:
    series = pd.Series([1, 2])
    assert PandasSeriesSummarizer().summarize(series, SHOW_DATA, compact=False) == repr(series)


@pandas_available
def test_pandas_dataframe_summarizer_summarize_max_items_none_shows_all_columns() -> None:
    frame = pd.DataFrame({f"c{i}": [i] for i in range(7)})
    result = PandasDataFrameSummarizer().summarize(
        frame, SummaryConfig(max_items=None), compact=False
    )
    assert result.endswith("c5: int64, c6: int64")


@pandas_available
def test_pandas_dataframe_summarizer_summarize_max_items_zero_hides_all_columns() -> None:
    frame = pd.DataFrame({"a": [1], "b": [2]})
    result = PandasDataFrameSummarizer().summarize(frame, SummaryConfig(max_items=0), compact=False)
    assert result == "<class 'pandas.DataFrame'> | shape=(1, 2) | columns=... (+2 more)"


@pandas_available
def test_pandas_dataframe_summarizer_summarize_compact_is_ignored() -> None:
    frame = pd.DataFrame({"a": [1]})
    summarizer = PandasDataFrameSummarizer()
    assert summarizer.summarize(frame, CONFIG, compact=True) == summarizer.summarize(
        frame, CONFIG, compact=False
    )
