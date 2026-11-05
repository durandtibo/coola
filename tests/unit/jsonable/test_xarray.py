from __future__ import annotations

from unittest.mock import patch

from coola.jsonable import to_jsonable
from coola.jsonable.xarray import get_default_transformers
from coola.testing.marks import xarray_available
from coola.utils.imports import is_xarray_available

if is_xarray_available():
    import xarray as xr


@xarray_available
def test_get_default_transformers_keys() -> None:
    assert list(get_default_transformers()) == [xr.DataArray, xr.Dataset, xr.Variable]


def test_get_default_transformers_without_xarray() -> None:
    with patch("coola.jsonable.xarray.is_xarray_available", lambda: False):
        assert get_default_transformers() == {}


@xarray_available
def test_to_jsonable_data_array() -> None:
    assert to_jsonable(xr.DataArray([1, 2], dims=["x"], name="a")) == {
        "dims": ["x"],
        "attrs": {},
        "data": [1, 2],
        "coords": {},
        "name": "a",
    }


@xarray_available
def test_to_jsonable_data_array_with_a_coordinate() -> None:
    result = to_jsonable(xr.DataArray([1, 2], dims=["x"], coords={"x": [10, 20]}))
    assert result["coords"] == {"x": {"dims": ["x"], "attrs": {}, "data": [10, 20]}}


@xarray_available
def test_to_jsonable_dataset() -> None:
    assert to_jsonable(xr.Dataset({"a": ("x", [1])})) == {
        "coords": {},
        "attrs": {},
        "dims": {"x": 1},
        "data_vars": {"a": {"dims": ["x"], "attrs": {}, "data": [1]}},
    }


@xarray_available
def test_to_jsonable_variable() -> None:
    assert to_jsonable(xr.Variable("x", [1])) == {"dims": ["x"], "attrs": {}, "data": [1]}


@xarray_available
def test_to_jsonable_data_array_in_a_container() -> None:
    assert to_jsonable([xr.Variable("x", [1])]) == [{"dims": ["x"], "attrs": {}, "data": [1]}]
