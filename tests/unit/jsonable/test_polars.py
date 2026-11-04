from __future__ import annotations

from unittest.mock import patch

import pytest

from coola.jsonable import to_jsonable
from coola.jsonable.polars import get_default_transformers
from coola.testing.marks import polars_available
from coola.utils.imports import is_polars_available

if is_polars_available():
    import polars as pl


@polars_available
def test_get_default_transformers_keys() -> None:
    assert list(get_default_transformers()) == [pl.Series]


def test_get_default_transformers_without_polars() -> None:
    with patch("coola.jsonable.polars.is_polars_available", lambda: False):
        assert get_default_transformers() == {}


@polars_available
def test_to_jsonable_series() -> None:
    assert to_jsonable(pl.Series([1, 2])) == [1, 2]


@polars_available
def test_to_jsonable_series_with_a_null() -> None:
    assert to_jsonable(pl.Series([1.5, None])) == [1.5, None]


@polars_available
def test_to_jsonable_series_empty() -> None:
    assert to_jsonable(pl.Series([], dtype=pl.Float64)) == []


@polars_available
def test_to_jsonable_series_of_strings() -> None:
    assert to_jsonable(pl.Series(["a", "b"])) == ["a", "b"]


@polars_available
def test_to_jsonable_series_in_a_container() -> None:
    assert to_jsonable({"a": pl.Series([1])}) == {"a": [1]}


@polars_available
def test_to_jsonable_data_frame_is_not_supported() -> None:
    with pytest.raises(TypeError, match=r"Cannot convert an object of type DataFrame"):
        to_jsonable(pl.DataFrame({"a": [1]}))
