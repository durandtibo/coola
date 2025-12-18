from __future__ import annotations

from types import ModuleType

import pytest

from coola.utils.fallback.pandas import pandas


def test_pandas_is_module_type() -> None:
    assert isinstance(pandas, ModuleType)


def test_pandas_module_name() -> None:
    assert pandas.__name__ == "pandas"


def test_pandas_dataframe_exists() -> None:
    assert hasattr(pandas, "DataFrame")


def test_pandas_dataframe_is_class() -> None:
    assert isinstance(pandas.DataFrame, type)


def test_pandas_dataframe_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'pandas' package is required but not installed."):
        pandas.DataFrame()


def test_pandas_series_exists() -> None:
    assert hasattr(pandas, "Series")


def test_pandas_series_is_class() -> None:
    assert isinstance(pandas.Series, type)


def test_pandas_series_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'pandas' package is required but not installed."):
        pandas.Series()
