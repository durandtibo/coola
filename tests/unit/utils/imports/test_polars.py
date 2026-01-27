from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    check_polars,
    is_polars_available,
    polars_available,
    raise_polars_missing_error,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_polars_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


##################
#     polars     #
##################


def test_check_polars_with_package() -> None:
    with patch("coola.utils.imports.polars.is_polars_available", lambda: True):
        check_polars()


def test_check_polars_without_package() -> None:
    with (
        patch("coola.utils.imports.polars.is_polars_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'polars' package is required but not installed."),
    ):
        check_polars()


def test_is_polars_available() -> None:
    assert isinstance(is_polars_available(), bool)


def test_polars_available_with_package() -> None:
    with patch("coola.utils.imports.polars.is_polars_available", lambda: True):
        fn = polars_available(my_function)
        assert fn(2) == 44


def test_polars_available_without_package() -> None:
    with patch("coola.utils.imports.polars.is_polars_available", lambda: False):
        fn = polars_available(my_function)
        assert fn(2) is None


def test_polars_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.polars.is_polars_available", lambda: True):

        @polars_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_polars_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.polars.is_polars_available", lambda: False):

        @polars_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_polars_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'polars' package is required but not installed."):
        raise_polars_missing_error()
