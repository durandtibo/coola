from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    check_pyarrow,
    is_pyarrow_available,
    pyarrow_available,
    raise_pyarrow_missing_error,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_pyarrow_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


###################
#     pyarrow     #
###################


def test_check_pyarrow_with_package() -> None:
    with patch("coola.utils.imports.pyarrow.is_pyarrow_available", lambda: True):
        check_pyarrow()


def test_check_pyarrow_without_package() -> None:
    with (
        patch("coola.utils.imports.pyarrow.is_pyarrow_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'pyarrow' package is required but not installed."),
    ):
        check_pyarrow()


def test_is_pyarrow_available() -> None:
    assert isinstance(is_pyarrow_available(), bool)


def test_pyarrow_available_with_package() -> None:
    with patch("coola.utils.imports.pyarrow.is_pyarrow_available", lambda: True):
        fn = pyarrow_available(my_function)
        assert fn(2) == 44


def test_pyarrow_available_without_package() -> None:
    with patch("coola.utils.imports.pyarrow.is_pyarrow_available", lambda: False):
        fn = pyarrow_available(my_function)
        assert fn(2) is None


def test_pyarrow_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.pyarrow.is_pyarrow_available", lambda: True):

        @pyarrow_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_pyarrow_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.pyarrow.is_pyarrow_available", lambda: False):

        @pyarrow_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_pyarrow_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'pyarrow' package is required but not installed."):
        raise_pyarrow_missing_error()
