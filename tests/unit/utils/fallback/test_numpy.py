from __future__ import annotations

from types import ModuleType

import pytest

from coola.utils.fallback.numpy import numpy


def test_numpy_is_module_type() -> None:
    assert isinstance(numpy, ModuleType)


def test_numpy_module_name() -> None:
    assert numpy.__name__ == "numpy"


def test_numpy_nested_module_access() -> None:
    assert hasattr(numpy, "ma")


def test_numpy_masked_array_exists() -> None:
    assert hasattr(numpy.ma, "MaskedArray")


def test_numpy_masked_array_is_class() -> None:
    assert isinstance(numpy.ma.MaskedArray, type)


def test_numpy_masked_array_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'numpy' package is required but not installed."):
        numpy.ma.MaskedArray()


def test_numpy_ndarray_exists() -> None:
    assert hasattr(numpy, "ndarray")


def test_numpy_ndarray_is_class() -> None:
    assert isinstance(numpy.ndarray, type)


def test_numpy_ndarray_instantiation() -> None:
    with pytest.raises(RuntimeError, match=r"'numpy' package is required but not installed."):
        numpy.ndarray()
