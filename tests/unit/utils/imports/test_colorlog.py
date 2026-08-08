from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    check_colorlog,
    colorlog_available,
    is_colorlog_available,
    raise_colorlog_missing_error,
)

logger = logging.getLogger(__name__)

MODULE = "coola.utils.imports.colorlog"


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_colorlog_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


####################
#     colorlog     #
####################


def test_check_colorlog_with_package() -> None:
    with patch(f"{MODULE}.is_colorlog_available", lambda: True):
        check_colorlog()


def test_check_colorlog_without_package() -> None:
    with (
        patch(f"{MODULE}.is_colorlog_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'colorlog' package is required but not installed."),
    ):
        check_colorlog()


def test_is_colorlog_available() -> None:
    assert isinstance(is_colorlog_available(), bool)


def test_colorlog_available_with_package() -> None:
    with patch(f"{MODULE}.is_colorlog_available", lambda: True):
        fn = colorlog_available(my_function)
        assert fn(2) == 44


def test_colorlog_available_without_package() -> None:
    with patch(f"{MODULE}.is_colorlog_available", lambda: False):
        fn = colorlog_available(my_function)
        assert fn(2) is None


def test_colorlog_available_decorator_with_package() -> None:
    with patch(f"{MODULE}.is_colorlog_available", lambda: True):

        @colorlog_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_colorlog_available_decorator_without_package() -> None:
    with patch(f"{MODULE}.is_colorlog_available", lambda: False):

        @colorlog_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_colorlog_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'colorlog' package is required but not installed."):
        raise_colorlog_missing_error()
