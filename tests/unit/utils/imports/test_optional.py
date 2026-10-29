from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from coola.testing.marks import torch_available
from coola.utils.imports import (
    MissingPackageError,
    check_torch_numpy,
    clear_availability_caches,
    is_torch_numpy_available,
    module_available,
    optional,
)

if TYPE_CHECKING:
    from collections.abc import Generator

MODULE = "coola.utils.imports.optional"
PACKAGES = [
    "colorlog",
    "jax",
    "numpy",
    "packaging",
    "pandas",
    "polars",
    "pyarrow",
    "pydantic",
    "rich",
    "scipy",
    "torch",
    "xarray",
]


@pytest.fixture(autouse=True)
def _clear_caches() -> Generator[None]:
    clear_availability_caches()
    yield
    clear_availability_caches()


########################################
#     Tests for is_<package>_available #
########################################


@pytest.mark.parametrize("package", PACKAGES)
def test_is_package_available_is_module_available(package: str) -> None:
    assert getattr(optional, f"is_{package}_available")() == module_available(package)


@pytest.mark.parametrize("package", PACKAGES)
def test_is_package_available_true(package: str) -> None:
    with patch("coola.utils.imports.package.importlib.import_module"):
        assert getattr(optional, f"is_{package}_available")()


@pytest.mark.parametrize("package", PACKAGES)
def test_is_package_available_false(package: str) -> None:
    with patch("coola.utils.imports.package.importlib.import_module", side_effect=ImportError):
        assert not getattr(optional, f"is_{package}_available")()


@pytest.mark.parametrize("package", PACKAGES)
def test_is_package_available_is_cached(package: str) -> None:
    with patch("coola.utils.imports.package.importlib.import_module") as mock:
        getattr(optional, f"is_{package}_available")()
        getattr(optional, f"is_{package}_available")()
    assert mock.call_count == 1


@pytest.mark.parametrize("package", PACKAGES)
def test_is_package_available_cache_is_cleared(package: str) -> None:
    with patch("coola.utils.imports.package.importlib.import_module") as mock:
        getattr(optional, f"is_{package}_available")()
        clear_availability_caches()
        getattr(optional, f"is_{package}_available")()
    assert mock.call_count == 2


####################################
#     Tests for check_<package>    #
####################################


@pytest.mark.parametrize("package", PACKAGES)
def test_check_package_available(package: str) -> None:
    with patch("coola.utils.imports.package.importlib.import_module"):
        getattr(optional, f"check_{package}")()


@pytest.mark.parametrize("package", PACKAGES)
def test_check_package_missing_message(package: str) -> None:
    msg = (
        f"'{package}' package is required but not installed. "
        f"You can install '{package}' package with the command:\n\n"
        f"pip install {package}\n\nor\n\n"
        f"uv pip install {package}\n"
    )
    with (
        patch("coola.utils.imports.package.importlib.import_module", side_effect=ImportError),
        pytest.raises(MissingPackageError, match="package is required but not installed") as info,
    ):
        getattr(optional, f"check_{package}")()
    assert str(info.value) == msg


@pytest.mark.parametrize("package", PACKAGES)
def test_check_package_missing_name(package: str) -> None:
    with (
        patch("coola.utils.imports.package.importlib.import_module", side_effect=ImportError),
        pytest.raises(MissingPackageError, match="package is required but not installed") as info,
    ):
        getattr(optional, f"check_{package}")()
    assert info.value.name == package


######################################
#     Tests for the torch+numpy pair #
######################################


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
        pytest.raises(MissingPackageError, match="packages are required") as exc_info,
    ):
        check_torch_numpy()
    assert exc_info.value.name == "torch"


def test_check_torch_numpy_missing_numpy_error_name() -> None:
    with (
        patch(f"{MODULE}.is_torch_numpy_available", lambda: False),
        patch(f"{MODULE}.is_torch_available", lambda: True),
        patch(f"{MODULE}.is_numpy_available", lambda: False),
        pytest.raises(MissingPackageError, match="packages are required") as exc_info,
    ):
        check_torch_numpy()
    assert exc_info.value.name == "numpy"


def test_check_torch_numpy_message_has_no_trailing_space() -> None:
    with (
        patch(f"{MODULE}.is_torch_numpy_available", lambda: False),
        pytest.raises(MissingPackageError, match="packages are required") as exc_info,
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
