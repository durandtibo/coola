from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola.utils.imports import (
    check_torch_numpy,
    is_torch_numpy_available,
    torch_numpy_available,
)

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> None:
    is_torch_numpy_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


#######################
#     torch+numpy     #
#######################


def test_check_torch_numpy() -> None:
    with patch("coola.utils.imports.torch_numpy.is_torch_numpy_available", lambda: True):
        check_torch_numpy()


def test_check_torch_numpy_missing() -> None:
    with (
        patch("coola.utils.imports.torch_numpy.is_torch_numpy_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'torch' and 'numpy' packages are required"),
    ):
        check_torch_numpy()


def test_is_torch_numpy_available() -> None:
    assert isinstance(is_torch_numpy_available(), bool)


def test_is_torch_numpy_available_missing_torch() -> None:
    with patch("coola.utils.imports.torch_numpy.is_torch_available", lambda: False):
        assert not is_torch_numpy_available()


def test_is_torch_numpy_available_missing_numpy() -> None:
    with (
        patch("coola.utils.imports.torch_numpy.is_torch_available", lambda: True),
        patch("coola.utils.imports.torch_numpy.is_numpy_available", lambda: False),
    ):
        assert not is_torch_numpy_available()


def test_is_torch_numpy_available_not_compatible() -> None:
    with (
        patch("coola.utils.imports.torch_numpy.is_torch_available", lambda: True),
        patch("coola.utils.imports.torch_numpy.is_numpy_available", lambda: True),
        patch("coola.utils.imports.torch_numpy.torch.tensor", Mock(side_effect=RuntimeError)),
    ):
        assert not is_torch_numpy_available()


def test_torch_numpy_available_with_package() -> None:
    with patch("coola.utils.imports.torch_numpy.is_torch_numpy_available", lambda: True):
        fn = torch_numpy_available(my_function)
        assert fn(2) == 44


def test_torch_numpy_available_without_package() -> None:
    with patch("coola.utils.imports.torch_numpy.is_torch_numpy_available", lambda: False):
        fn = torch_numpy_available(my_function)
        assert fn(2) is None


def test_torch_numpy_available_decorator_with_package() -> None:
    with patch("coola.utils.imports.torch_numpy.is_torch_numpy_available", lambda: True):

        @torch_numpy_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_torch_numpy_available_decorator_without_package() -> None:
    with patch("coola.utils.imports.torch_numpy.is_torch_numpy_available", lambda: False):

        @torch_numpy_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None
