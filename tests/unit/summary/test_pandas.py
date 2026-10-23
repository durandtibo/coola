from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from coola.summary import (
    PandasDataFrameSummarizer,
    PandasSeriesSummarizer,
    SummarizerRegistry,
)
from coola.testing.fixtures import pandas_available
from coola.utils.imports import MissingPackageError, is_pandas_available

if is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    pd = Mock()


@pytest.fixture
def registry() -> SummarizerRegistry:
    return SummarizerRegistry()


SUMMARIZERS = [
    PandasDataFrameSummarizer,
    PandasSeriesSummarizer,
]


@pandas_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_init_default(summarizer_cls: type) -> None:
    assert summarizer_cls()._show_data is False


@pandas_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_init_show_data(summarizer_cls: type) -> None:
    assert summarizer_cls(show_data=True)._show_data is True


@pandas_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_repr_default(summarizer_cls: type) -> None:
    assert repr(summarizer_cls()) == f"{summarizer_cls.__name__}(show_data=False)"


@pandas_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_repr_show_data(summarizer_cls: type) -> None:
    assert repr(summarizer_cls(show_data=True)) == f"{summarizer_cls.__name__}(show_data=True)"


@pandas_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_str(summarizer_cls: type) -> None:
    assert str(summarizer_cls()) == f"{summarizer_cls.__name__}(show_data=False)"


@pandas_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_true_default(summarizer_cls: type) -> None:
    assert summarizer_cls().equal(summarizer_cls())


@pandas_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_true_show_data(summarizer_cls: type) -> None:
    assert summarizer_cls(show_data=True).equal(summarizer_cls(show_data=True))


@pandas_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_false_different_show_data(summarizer_cls: type) -> None:
    assert not summarizer_cls(show_data=False).equal(summarizer_cls(show_data=True))


@pandas_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_false_different_type(summarizer_cls: type) -> None:
    assert not summarizer_cls().equal(42)


@pandas_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_false_child_class(summarizer_cls: type) -> None:
    class Child(summarizer_cls): ...

    assert not summarizer_cls().equal(Child())


@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_missing_package_raises(summarizer_cls: type) -> None:
    with (
        patch("coola.utils.imports.pandas.is_pandas_available", lambda: False),
        pytest.raises(MissingPackageError, match="'pandas' package is required"),
    ):
        summarizer_cls()


@pandas_available
def test_pandas_dataframe_summarizer_summarize(registry: SummarizerRegistry) -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3], "c": ["x", "y", "z"]})
    result = PandasDataFrameSummarizer().summarize(frame, registry)
    assert result.startswith(f"{type(frame)} | shape=(3, 3) | columns=a: int64, b: float64, c: ")


@pandas_available
def test_pandas_dataframe_summarizer_summarize_empty(registry: SummarizerRegistry) -> None:
    frame = pd.DataFrame()
    result = PandasDataFrameSummarizer().summarize(frame, registry)
    assert result == f"{type(frame)} | shape=(0, 0) | columns="


@pandas_available
def test_pandas_dataframe_summarizer_summarize_class_name_class_prefix(
    registry: SummarizerRegistry,
) -> None:
    result = PandasDataFrameSummarizer().summarize(pd.DataFrame({"a": [1]}), registry)
    assert "<class 'pandas." in result


@pandas_available
def test_pandas_dataframe_summarizer_summarize_class_name_class_name(
    registry: SummarizerRegistry,
) -> None:
    result = PandasDataFrameSummarizer().summarize(pd.DataFrame({"a": [1]}), registry)
    assert ".DataFrame'>" in result


@pandas_available
def test_pandas_dataframe_summarizer_summarize_many_columns_shows_the_first_columns(
    registry: SummarizerRegistry,
) -> None:
    frame = pd.DataFrame({f"c{i}": [i] for i in range(13)})
    result = PandasDataFrameSummarizer().summarize(frame, registry)
    assert "c9: int64, ... (+3 more)" in result


@pandas_available
def test_pandas_dataframe_summarizer_summarize_many_columns_hides_the_other_columns(
    registry: SummarizerRegistry,
) -> None:
    frame = pd.DataFrame({f"c{i}": [i] for i in range(13)})
    result = PandasDataFrameSummarizer().summarize(frame, registry)
    assert "c10" not in result


@pandas_available
def test_pandas_dataframe_summarizer_summarize_show_data(registry: SummarizerRegistry) -> None:
    frame = pd.DataFrame({"a": [1, 2]})
    assert PandasDataFrameSummarizer(show_data=True).summarize(frame, registry) == repr(frame)


@pandas_available
def test_pandas_series_summarizer_summarize(registry: SummarizerRegistry) -> None:
    series = pd.Series([1, 2, 3], name="a")
    result = PandasSeriesSummarizer().summarize(series, registry)
    assert result == f"{type(series)} | shape=(3,) | dtype=int64 | name='a'"


@pandas_available
def test_pandas_series_summarizer_summarize_class_name_class_prefix(
    registry: SummarizerRegistry,
) -> None:
    result = PandasSeriesSummarizer().summarize(pd.Series([1]), registry)
    assert "<class 'pandas." in result


@pandas_available
def test_pandas_series_summarizer_summarize_class_name_class_name(
    registry: SummarizerRegistry,
) -> None:
    result = PandasSeriesSummarizer().summarize(pd.Series([1]), registry)
    assert ".Series'>" in result


@pandas_available
def test_pandas_series_summarizer_summarize_no_name(registry: SummarizerRegistry) -> None:
    series = pd.Series([1.0, 2.0])
    result = PandasSeriesSummarizer().summarize(series, registry)
    assert result == f"{type(series)} | shape=(2,) | dtype=float64 | name=None"


@pandas_available
def test_pandas_series_summarizer_summarize_show_data(registry: SummarizerRegistry) -> None:
    series = pd.Series([1, 2])
    assert PandasSeriesSummarizer(show_data=True).summarize(series, registry) == repr(series)


@pandas_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_depth_and_max_depth_ignored(
    summarizer_cls: type, registry: SummarizerRegistry
) -> None:
    data = (
        pd.DataFrame({"a": [1]}) if summarizer_cls is PandasDataFrameSummarizer else pd.Series([1])
    )
    summarizer = summarizer_cls()
    expected = summarizer.summarize(data, registry)
    assert summarizer.summarize(data, registry, depth=5, max_depth=10) == expected


@pandas_available
def test_dataframe_summarizer_max_fields(registry: SummarizerRegistry) -> None:
    df = pd.DataFrame({"a": [1], "b": [2.0], "c": ["x"]})
    out = PandasDataFrameSummarizer(max_fields=2).summarize(df, registry)
    assert out.endswith("columns=a: int64, b: float64, ... (+1 more)")


@pandas_available
def test_dataframe_summarizer_max_fields_repr() -> None:
    assert repr(PandasDataFrameSummarizer(max_fields=2)) == (
        "PandasDataFrameSummarizer(show_data=False, max_fields=2)"
    )
