from __future__ import annotations

from unittest.mock import patch

from coola.jsonable import to_jsonable
from coola.jsonable.pandas import get_default_transformers
from coola.testing.marks import pandas_available
from coola.utils.imports import is_pandas_available

if is_pandas_available():
    import pandas as pd


@pandas_available
def test_get_default_transformers_keys() -> None:
    assert list(get_default_transformers()) == [pd.Series, pd.Index, type(pd.NaT), type(pd.NA)]


def test_get_default_transformers_without_pandas() -> None:
    with patch("coola.jsonable.pandas.is_pandas_available", lambda: False):
        assert get_default_transformers() == {}


@pandas_available
def test_to_jsonable_series() -> None:
    assert to_jsonable(pd.Series([1, 2])) == [1, 2]


@pandas_available
def test_to_jsonable_series_empty() -> None:
    assert to_jsonable(pd.Series([], dtype=float)) == []


@pandas_available
def test_to_jsonable_series_of_strings() -> None:
    assert to_jsonable(pd.Series(["a", "b"])) == ["a", "b"]


@pandas_available
def test_to_jsonable_series_of_timestamps_with_a_missing_value() -> None:
    assert to_jsonable(pd.Series(pd.to_datetime(["2020-01-01", None]))) == [
        "2020-01-01T00:00:00",
        None,
    ]


@pandas_available
def test_to_jsonable_index() -> None:
    assert to_jsonable(pd.Index(["a", "b"])) == ["a", "b"]


@pandas_available
def test_to_jsonable_nat() -> None:
    assert to_jsonable(pd.NaT) is None


@pandas_available
def test_to_jsonable_na() -> None:
    assert to_jsonable(pd.NA) is None


@pandas_available
def test_to_jsonable_series_in_a_container() -> None:
    assert to_jsonable({"a": pd.Series([1])}) == {"a": [1]}


@pandas_available
def test_to_jsonable_timestamp() -> None:
    assert to_jsonable(pd.Timestamp("2020-01-01")) == "2020-01-01T00:00:00"
