from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from coola.summary import (
    PolarsDataFrameSummarizer,
    PolarsLazyFrameSummarizer,
    PolarsSeriesSummarizer,
    SummarizerRegistry,
)
from coola.testing.fixtures import polars_available
from coola.utils.imports import MissingPackageError, is_polars_available

if is_polars_available():
    import polars as pl
else:  # pragma: no cover
    pl = Mock()


@pytest.fixture
def registry() -> SummarizerRegistry:
    return SummarizerRegistry()


SUMMARIZERS = [
    PolarsDataFrameSummarizer,
    PolarsLazyFrameSummarizer,
    PolarsSeriesSummarizer,
]


@polars_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_init_default(summarizer_cls: type) -> None:
    assert summarizer_cls()._show_data is False


@polars_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_init_show_data(summarizer_cls: type) -> None:
    assert summarizer_cls(show_data=True)._show_data is True


@polars_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_repr_default(summarizer_cls: type) -> None:
    assert repr(summarizer_cls()) == f"{summarizer_cls.__name__}(show_data=False)"


@polars_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_repr_show_data(summarizer_cls: type) -> None:
    assert repr(summarizer_cls(show_data=True)) == f"{summarizer_cls.__name__}(show_data=True)"


@polars_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_str(summarizer_cls: type) -> None:
    assert str(summarizer_cls()) == f"{summarizer_cls.__name__}(show_data=False)"


@polars_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_true_default(summarizer_cls: type) -> None:
    assert summarizer_cls().equal(summarizer_cls())


@polars_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_true_show_data(summarizer_cls: type) -> None:
    assert summarizer_cls(show_data=True).equal(summarizer_cls(show_data=True))


@polars_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_false_different_show_data(summarizer_cls: type) -> None:
    assert not summarizer_cls(show_data=False).equal(summarizer_cls(show_data=True))


@polars_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_false_different_type(summarizer_cls: type) -> None:
    assert not summarizer_cls().equal(42)


@polars_available
@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_equal_false_child_class(summarizer_cls: type) -> None:
    class Child(summarizer_cls): ...

    assert not summarizer_cls().equal(Child())


@pytest.mark.parametrize("summarizer_cls", SUMMARIZERS)
def test_summarizer_missing_package_raises(summarizer_cls: type) -> None:
    with (
        patch("coola.utils.imports.polars.is_polars_available", lambda: False),
        pytest.raises(MissingPackageError, match="'polars' package is required"),
    ):
        summarizer_cls()


@polars_available
def test_polars_dataframe_summarizer_summarize_columns(registry: SummarizerRegistry) -> None:
    frame = pl.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    result = PolarsDataFrameSummarizer().summarize(frame, registry)
    assert result.endswith(" | shape=(3, 2) | columns=a: Int64, b: Float64")


@polars_available
def test_polars_dataframe_summarizer_summarize_class_name(registry: SummarizerRegistry) -> None:
    frame = pl.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    result = PolarsDataFrameSummarizer().summarize(frame, registry)
    assert "DataFrame" in result.split(" | ")[0]


@polars_available
def test_polars_dataframe_summarizer_summarize_many_columns(registry: SummarizerRegistry) -> None:
    frame = pl.DataFrame({f"c{i}": [i] for i in range(13)})
    result = PolarsDataFrameSummarizer().summarize(frame, registry)
    assert result.endswith("c9: Int64, ... (+3 more)")


@polars_available
def test_polars_dataframe_summarizer_summarize_show_data(registry: SummarizerRegistry) -> None:
    frame = pl.DataFrame({"a": [1, 2]})
    assert PolarsDataFrameSummarizer(show_data=True).summarize(frame, registry) == repr(frame)


@polars_available
def test_polars_lazyframe_summarizer_summarize_does_not_collect(
    registry: SummarizerRegistry,
) -> None:
    frame = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    with patch.object(pl.LazyFrame, "collect", side_effect=AssertionError("collected")):
        result = PolarsLazyFrameSummarizer().summarize(frame, registry)
    assert result.endswith(" | columns=a: Int64, b: String")


@polars_available
def test_polars_lazyframe_summarizer_summarize_class_name(registry: SummarizerRegistry) -> None:
    frame = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = PolarsLazyFrameSummarizer().summarize(frame, registry)
    assert "LazyFrame" in result.split(" | ")[0]


@polars_available
def test_polars_lazyframe_summarizer_summarize_invalid_plan(registry: SummarizerRegistry) -> None:
    frame = pl.LazyFrame({"a": [1, 2, 3]}).select("missing")
    result = PolarsLazyFrameSummarizer().summarize(frame, registry)
    assert result.endswith(" | columns=<unresolved schema: ColumnNotFoundError>")


@polars_available
def test_polars_lazyframe_summarizer_summarize_show_data(registry: SummarizerRegistry) -> None:
    frame = pl.LazyFrame({"a": [1, 2]})
    assert PolarsLazyFrameSummarizer(show_data=True).summarize(frame, registry) == repr(frame)


@polars_available
def test_polars_series_summarizer_summarize_columns(registry: SummarizerRegistry) -> None:
    result = PolarsSeriesSummarizer().summarize(pl.Series("a", [1, 2, 3]), registry)
    assert result.endswith(" | shape=(3,) | dtype=Int64 | name='a'")


@polars_available
def test_polars_series_summarizer_summarize_class_name(registry: SummarizerRegistry) -> None:
    result = PolarsSeriesSummarizer().summarize(pl.Series("a", [1, 2, 3]), registry)
    assert "Series" in result.split(" | ")[0]


@polars_available
def test_polars_series_summarizer_summarize_show_data(registry: SummarizerRegistry) -> None:
    series = pl.Series("a", [1, 2])
    assert PolarsSeriesSummarizer(show_data=True).summarize(series, registry) == repr(series)


@polars_available
def test_dataframe_summarizer_max_fields(registry: SummarizerRegistry) -> None:
    df = pl.DataFrame({"a": [1], "b": [2.0], "c": ["x"]})
    out = PolarsDataFrameSummarizer(max_fields=1).summarize(df, registry)
    assert out.endswith("columns=a: Int64, ... (+2 more)")


@polars_available
def test_lazyframe_summarizer_max_fields(registry: SummarizerRegistry) -> None:
    df = pl.DataFrame({"a": [1], "b": [2.0], "c": ["x"]})
    out = PolarsLazyFrameSummarizer(max_fields=1).summarize(df.lazy(), registry)
    assert out.endswith("columns=a: Int64, ... (+2 more)")
