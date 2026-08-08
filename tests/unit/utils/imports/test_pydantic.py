from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    check_pydantic,
    is_pydantic_available,
    pydantic_available,
    raise_pydantic_missing_error,
)

logger = logging.getLogger(__name__)


MODULE = "coola.utils.imports.pydantic"


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_pydantic_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


####################
#     pydantic     #
####################


def test_check_pydantic_with_package() -> None:
    with patch(f"{MODULE}.is_pydantic_available", lambda: True):
        check_pydantic()


def test_check_pydantic_without_package() -> None:
    with (
        patch(f"{MODULE}.is_pydantic_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'pydantic' package is required but not installed."),
    ):
        check_pydantic()


def test_is_pydantic_available() -> None:
    assert isinstance(is_pydantic_available(), bool)


def test_pydantic_available_with_package() -> None:
    with patch(f"{MODULE}.is_pydantic_available", lambda: True):
        fn = pydantic_available(my_function)
        assert fn(2) == 44


def test_pydantic_available_without_package() -> None:
    with patch(f"{MODULE}.is_pydantic_available", lambda: False):
        fn = pydantic_available(my_function)
        assert fn(2) is None


def test_pydantic_available_decorator_with_package() -> None:
    with patch(f"{MODULE}.is_pydantic_available", lambda: True):

        @pydantic_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_pydantic_available_decorator_without_package() -> None:
    with patch(f"{MODULE}.is_pydantic_available", lambda: False):

        @pydantic_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_pydantic_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'pydantic' package is required but not installed."):
        raise_pydantic_missing_error()
