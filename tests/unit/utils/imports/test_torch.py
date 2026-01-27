from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    check_torch,
    is_torch_available,
    raise_torch_missing_error,
    torch_available,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_torch_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


#################
#     torch     #
#################


def test_check_torch_with_package() -> None:
    with patch("coola.utils.imports.torch.is_torch_available", lambda: True):
        check_torch()


def test_check_torch_without_package() -> None:
    with (
        patch("coola.utils.imports.torch.is_torch_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."),
    ):
        check_torch()


def test_is_torch_available() -> None:
    assert isinstance(is_torch_available(), bool)


def test_torch_available_with_package() -> None:
    with patch("coola.utils.imports.torch.is_torch_available", lambda: True):
        fn = torch_available(my_function)
        assert fn(2) == 44


def test_torch_available_without_package() -> None:
    with patch("coola.utils.imports.torch.is_torch_available", lambda: False):
        fn = torch_available(my_function)
        assert fn(2) is None


def test_torch_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.torch.is_torch_available", lambda: True):

        @torch_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_torch_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.torch.is_torch_available", lambda: False):

        @torch_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


def test_raise_torch_missing_error() -> None:
    with pytest.raises(RuntimeError, match=r"'torch' package is required but not installed."):
        raise_torch_missing_error()
