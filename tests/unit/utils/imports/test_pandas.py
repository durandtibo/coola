from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    check_pandas,
    is_pandas_available,
    pandas_available,
    raise_pandas_missing_error,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_pandas_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


##################
#     pandas     #
##################


def test_check_pandas_with_package() -> None:
    with patch("coola.utils.imports.pandas.is_pandas_available", lambda: True):
        check_pandas()


def test_check_pandas_without_package() -> None:
    with (
        patch("coola.utils.imports.pandas.is_pandas_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'pandas' package is required but not installed."),
    ):
        check_pandas()


def test_is_pandas_available() -> None:
    assert isinstance(is_pandas_available(), bool)


def test_pandas_available_with_package() -> None:
    with patch("coola.utils.imports.pandas.is_pandas_available", lambda: True):
        fn = pandas_available(my_function)
        assert fn(2) == 44


def test_pandas_available_without_package() -> None:
    with patch("coola.utils.imports.pandas.is_pandas_available", lambda: False):
        fn = pandas_available(my_function)
        assert fn(2) is None


def test_pandas_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.pandas.is_pandas_available", lambda: True):

        @pandas_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_pandas_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.pandas.is_pandas_available", lambda: False):

        @pandas_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_pandas_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'pandas' package is required but not installed."):
        raise_pandas_missing_error()
