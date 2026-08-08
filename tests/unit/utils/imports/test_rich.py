from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    check_rich,
    is_rich_available,
    raise_rich_missing_error,
    rich_available,
)

logger = logging.getLogger(__name__)


MODULE = "coola.utils.imports.rich"


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_rich_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


################
#     rich     #
################


def test_check_rich_with_package() -> None:
    with patch(f"{MODULE}.is_rich_available", lambda: True):
        check_rich()


def test_check_rich_without_package() -> None:
    with (
        patch(f"{MODULE}.is_rich_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'rich' package is required but not installed."),
    ):
        check_rich()


def test_is_rich_available() -> None:
    assert isinstance(is_rich_available(), bool)


def test_rich_available_with_package() -> None:
    with patch(f"{MODULE}.is_rich_available", lambda: True):
        fn = rich_available(my_function)
        assert fn(2) == 44


def test_rich_available_without_package() -> None:
    with patch(f"{MODULE}.is_rich_available", lambda: False):
        fn = rich_available(my_function)
        assert fn(2) is None


def test_rich_available_decorator_with_package() -> None:
    with patch(f"{MODULE}.is_rich_available", lambda: True):

        @rich_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_rich_available_decorator_without_package() -> None:
    with patch(f"{MODULE}.is_rich_available", lambda: False):

        @rich_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_rich_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'rich' package is required but not installed."):
        raise_rich_missing_error()
