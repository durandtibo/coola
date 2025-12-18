from __future__ import annotations

from types import ModuleType

import pytest

from coola.utils.fallback.xarray import xarray


def test_xarray_is_module_type() -> None:
    assert isinstance(xarray, ModuleType)


def test_xarray_module_name() -> None:
    assert xarray.__name__ == "xarray"


def test_xarray_data_array_exists() -> None:
    assert hasattr(xarray, "DataArray")


def test_xarray_data_array_is_class() -> None:
    assert isinstance(xarray.DataArray, type)


def test_xarray_data_array_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'xarray' package is required but not installed."):
        xarray.DataArray()


def test_xarray_dataset_exists() -> None:
    assert hasattr(xarray, "Dataset")


def test_xarray_dataset_is_class() -> None:
    assert isinstance(xarray.Dataset, type)


def test_xarray_dataset_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'xarray' package is required but not installed."):
        xarray.Dataset()


def test_xarray_variable_exists() -> None:
    assert hasattr(xarray, "Variable")


def test_xarray_variable_is_class() -> None:
    assert isinstance(xarray.Variable, type)


def test_xarray_variable_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'xarray' package is required but not installed."):
        xarray.Variable()
