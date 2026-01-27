from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    check_packaging,
    is_packaging_available,
    packaging_available,
    raise_packaging_missing_error,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_packaging_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


#####################
#     packaging     #
#####################


def test_check_packaging_with_package() -> None:
    with patch("coola.utils.imports.packaging.is_packaging_available", lambda: True):
        check_packaging()


def test_check_packaging_without_package() -> None:
    with (
        patch("coola.utils.imports.packaging.is_packaging_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'packaging' package is required but not installed."),
    ):
        check_packaging()


def test_is_packaging_available() -> None:
    assert isinstance(is_packaging_available(), bool)


def test_packaging_available_with_package() -> None:
    with patch("coola.utils.imports.packaging.is_packaging_available", lambda: True):
        fn = packaging_available(my_function)
        assert fn(2) == 44


def test_packaging_available_without_package() -> None:
    with patch("coola.utils.imports.packaging.is_packaging_available", lambda: False):
        fn = packaging_available(my_function)
        assert fn(2) is None


def test_packaging_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.packaging.is_packaging_available", lambda: True):

        @packaging_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_packaging_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.packaging.is_packaging_available", lambda: False):

        @packaging_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_packaging_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'packaging' package is required but not installed."):
        raise_packaging_missing_error()
