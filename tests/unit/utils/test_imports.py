from __future__ import annotations

from unittest.mock import patch

from pytest import raises

from coola.utils.imports import (
    check_numpy,
    check_pandas,
    check_polars,
    check_torch,
    check_xarray,
    is_numpy_available,
    is_pandas_available,
    is_polars_available,
    is_torch_available,
    is_xarray_available,
)


def test_check_numpy_with_package() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda *args: True):
        check_numpy()


def test_check_numpy_without_package() -> None:
    with patch("coola.utils.imports.is_numpy_available", lambda *args: False):
        with raises(RuntimeError, match="`numpy` package is required but not installed."):
            check_numpy()


def test_is_numpy_available() -> None:
    assert isinstance(is_numpy_available(), bool)


def test_check_pandas_with_package() -> None:
    with patch("coola.utils.imports.is_pandas_available", lambda *args: True):
        check_pandas()


def test_check_pandas_without_package() -> None:
    with patch("coola.utils.imports.is_pandas_available", lambda *args: False):
        with raises(RuntimeError, match="`pandas` package is required but not installed."):
            check_pandas()


def test_is_pandas_available() -> None:
    assert isinstance(is_pandas_available(), bool)


def test_check_polars_with_package() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda *args: True):
        check_polars()


def test_check_polars_without_package() -> None:
    with patch("coola.utils.imports.is_polars_available", lambda *args: False):
        with raises(RuntimeError, match="`polars` package is required but not installed."):
            check_polars()


def test_is_polars_available() -> None:
    assert isinstance(is_polars_available(), bool)


def test_check_torch_with_package() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda *args: True):
        check_torch()


def test_check_torch_without_package() -> None:
    with patch("coola.utils.imports.is_torch_available", lambda *args: False):
        with raises(RuntimeError, match="`torch` package is required but not installed."):
            check_torch()


def test_is_torch_available() -> None:
    assert isinstance(is_torch_available(), bool)


def test_check_xarray_with_package() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda *args: True):
        check_xarray()


def test_check_xarray_without_package() -> None:
    with patch("coola.utils.imports.is_xarray_available", lambda *args: False):
        with raises(RuntimeError, match="`xarray` package is required but not installed."):
            check_xarray()


def test_is_xarray_available() -> None:
    assert isinstance(is_xarray_available(), bool)
