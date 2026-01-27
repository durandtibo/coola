from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    check_xarray,
    is_xarray_available,
    raise_xarray_missing_error,
    xarray_available,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_xarray_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


##################
#     xarray     #
##################


def test_check_xarray_with_package() -> None:
    with patch("coola.utils.imports.xarray.is_xarray_available", lambda: True):
        check_xarray()


def test_check_xarray_without_package() -> None:
    with (
        patch("coola.utils.imports.xarray.is_xarray_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'xarray' package is required but not installed."),
    ):
        check_xarray()


def test_is_xarray_available() -> None:
    assert isinstance(is_xarray_available(), bool)


def test_xarray_available_with_package() -> None:
    with patch("coola.utils.imports.xarray.is_xarray_available", lambda: True):
        fn = xarray_available(my_function)
        assert fn(2) == 44


def test_xarray_available_without_package() -> None:
    with patch("coola.utils.imports.xarray.is_xarray_available", lambda: False):
        fn = xarray_available(my_function)
        assert fn(2) is None


def test_xarray_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.xarray.is_xarray_available", lambda: True):

        @xarray_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_xarray_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.xarray.is_xarray_available", lambda: False):

        @xarray_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_xarray_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'xarray' package is required but not installed."):
        raise_xarray_missing_error()
