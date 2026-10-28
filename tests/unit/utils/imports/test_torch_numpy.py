from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from coola.testing.marks import torch_available
from coola.utils.imports import (
    MissingPackageError,
    check_torch_numpy,
    is_torch_numpy_available,
    torch_numpy_available,
)

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

MODULE = "coola.utils.imports.torch_numpy"


@pytest.fixture(autouse=True)
def _cache_clear() -> Generator[None]:
    is_torch_numpy_available.cache_clear()
    yield
    is_torch_numpy_available.cache_clear()


def my_function(n: int = 0) -> int:
    return 42 + n


#######################
#     torch+numpy     #
#######################


def test_check_torch_numpy() -> None:
    with patch(f"{MODULE}.is_torch_numpy_available", lambda: True):
        check_torch_numpy()


def test_check_torch_numpy_missing() -> None:
    with (
        patch(f"{MODULE}.is_torch_numpy_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'torch' and 'numpy' packages are required"),
    ):
        check_torch_numpy()


def test_check_torch_numpy_missing_error_name() -> None:
    with (
        patch(f"{MODULE}.is_torch_numpy_available", lambda: False),
        patch(f"{MODULE}.is_torch_available", lambda: False),
        pytest.raises(MissingPackageError) as exc_info,
    ):
        check_torch_numpy()
    assert exc_info.value.name == "torch"


def test_check_torch_numpy_missing_numpy_error_name() -> None:
    with (
        patch(f"{MODULE}.is_torch_numpy_available", lambda: False),
        patch(f"{MODULE}.is_torch_available", lambda: True),
        patch(f"{MODULE}.is_numpy_available", lambda: False),
        pytest.raises(MissingPackageError) as exc_info,
    ):
        check_torch_numpy()
    assert exc_info.value.name == "numpy"


def test_check_torch_numpy_message_has_no_trailing_space() -> None:
    with (
        patch(f"{MODULE}.is_torch_numpy_available", lambda: False),
        pytest.raises(MissingPackageError) as exc_info,
    ):
        check_torch_numpy()
    assert not str(exc_info.value).endswith(" ")


def test_is_torch_numpy_available() -> None:
    assert isinstance(is_torch_numpy_available(), bool)


def test_is_torch_numpy_available_missing_torch() -> None:
    with patch(f"{MODULE}.is_torch_available", lambda: False):
        assert not is_torch_numpy_available()


def test_is_torch_numpy_available_missing_numpy() -> None:
    with (
        patch(f"{MODULE}.is_torch_available", lambda: True),
        patch(f"{MODULE}.is_numpy_available", lambda: False),
    ):
        assert not is_torch_numpy_available()


@torch_available
def test_is_torch_numpy_available_not_compatible() -> None:
    with (
        patch(f"{MODULE}.is_torch_available", lambda: True),
        patch(f"{MODULE}.is_numpy_available", lambda: True),
        patch("torch.tensor", Mock(side_effect=RuntimeError)),
    ):
        assert not is_torch_numpy_available()


@torch_available
def test_is_torch_numpy_available_compatible() -> None:
    with (
        patch(f"{MODULE}.is_torch_available", lambda: True),
        patch(f"{MODULE}.is_numpy_available", lambda: True),
        patch("torch.tensor", Mock(return_value=Mock(numpy=Mock(return_value=[1.0])))),
    ):
        assert is_torch_numpy_available()


def test_torch_numpy_available_with_package() -> None:
    with patch(f"{MODULE}.is_torch_numpy_available", lambda: True):
        fn = torch_numpy_available(my_function)
        assert fn(2) == 44


def test_torch_numpy_available_without_package() -> None:
    with patch(f"{MODULE}.is_torch_numpy_available", lambda: False):
        fn = torch_numpy_available(my_function)
        assert fn(2) is None


def test_torch_numpy_available_decorator_with_package() -> None:
    with patch(f"{MODULE}.is_torch_numpy_available", lambda: True):

        @torch_numpy_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_torch_numpy_available_decorator_without_package() -> None:
    with patch(f"{MODULE}.is_torch_numpy_available", lambda: False):

        @torch_numpy_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None
