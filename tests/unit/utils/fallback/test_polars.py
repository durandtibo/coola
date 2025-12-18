from __future__ import annotations

from types import ModuleType

import pytest

from coola.utils.fallback.polars import polars


def test_polars_is_module_type() -> None:
    assert isinstance(polars, ModuleType)


def test_polars_module_name() -> None:
    assert polars.__name__ == "polars"


def test_polars_dataframe_exists() -> None:
    assert hasattr(polars, "DataFrame")


def test_polars_dataframe_is_class() -> None:
    assert isinstance(polars.DataFrame, type)


def test_polars_dataframe_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'polars' package is required but not installed."):
        polars.DataFrame()


def test_polars_lazyframe_exists() -> None:
    assert hasattr(polars, "LazyFrame")


def test_polars_lazyframe_is_class() -> None:
    assert isinstance(polars.LazyFrame, type)


def test_polars_lazyframe_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'polars' package is required but not installed."):
        polars.LazyFrame()


def test_polars_series_exists() -> None:
    assert hasattr(polars, "Series")


def test_polars_series_is_class() -> None:
    assert isinstance(polars.Series, type)


def test_polars_series_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'polars' package is required but not installed."):
        polars.Series()
