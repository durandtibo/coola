from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    check_numpy,
    is_numpy_available,
    numpy_available,
    raise_numpy_missing_error,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_numpy_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


#################
#     numpy     #
#################


def test_check_numpy_with_package() -> None:
    with patch("coola.utils.imports.numpy.is_numpy_available", lambda: True):
        check_numpy()


def test_check_numpy_without_package() -> None:
    with (
        patch("coola.utils.imports.numpy.is_numpy_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'numpy' package is required but not installed."),
    ):
        check_numpy()


def test_is_numpy_available() -> None:
    assert isinstance(is_numpy_available(), bool)


def test_numpy_available_with_package() -> None:
    with patch("coola.utils.imports.numpy.is_numpy_available", lambda: True):
        fn = numpy_available(my_function)
        assert fn(2) == 44


def test_numpy_available_without_package() -> None:
    with patch("coola.utils.imports.numpy.is_numpy_available", lambda: False):
        fn = numpy_available(my_function)
        assert fn(2) is None


def test_numpy_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.numpy.is_numpy_available", lambda: True):

        @numpy_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_numpy_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.numpy.is_numpy_available", lambda: False):

        @numpy_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_numpy_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'numpy' package is required but not installed."):
        raise_numpy_missing_error()
