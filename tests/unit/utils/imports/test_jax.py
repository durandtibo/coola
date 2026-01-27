from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    check_jax,
    is_jax_available,
    jax_available,
    raise_jax_missing_error,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_jax_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


###############
#     jax     #
###############


def test_check_jax_with_package() -> None:
    with patch("coola.utils.imports.jax.is_jax_available", lambda: True):
        check_jax()


def test_check_jax_without_package() -> None:
    with (
        patch("coola.utils.imports.jax.is_jax_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'jax' package is required but not installed."),
    ):
        check_jax()


def test_is_jax_available() -> None:
    assert isinstance(is_jax_available(), bool)


def test_jax_available_with_package() -> None:
    with patch("coola.utils.imports.jax.is_jax_available", lambda: True):
        fn = jax_available(my_function)
        assert fn(2) == 44


def test_jax_available_without_package() -> None:
    with patch("coola.utils.imports.jax.is_jax_available", lambda: False):
        fn = jax_available(my_function)
        assert fn(2) is None


def test_jax_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.jax.is_jax_available", lambda: True):

        @jax_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_jax_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.jax.is_jax_available", lambda: False):

        @jax_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_jax_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'jax' package is required but not installed."):
        raise_jax_missing_error()
