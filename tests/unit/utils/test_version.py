from __future__ import annotations

import operator
from unittest.mock import patch

import pytest

from coola.testing.marks import packaging_available
from coola.utils.imports import (
    MissingPackageError,
    clear_availability_caches,
    is_packaging_available,
)
from coola.utils.version import compare_version, get_package_version

if is_packaging_available():
    from packaging.version import Version

#####################################
#     Tests for compare_version     #
#####################################


@packaging_available
def test_compare_version_true() -> None:
    assert compare_version("pytest", operator.ge, "7.3.0")


@packaging_available
def test_compare_version_false() -> None:
    assert not compare_version("pytest", operator.le, "7.3.0")


@packaging_available
def test_compare_version_false_missing() -> None:
    assert not compare_version("missing", operator.ge, "1.0.0")


def test_compare_version_missing_packaging() -> None:
    with (
        patch("coola.utils.imports.optional.is_packaging_available", lambda: False),
        pytest.raises(
            MissingPackageError, match=r"'packaging' package is required but not installed."
        ),
    ):
        compare_version("my_package", operator.ge, "7.3.0")


#########################################
#     Tests for get_package_version     #
#########################################


@packaging_available
@pytest.mark.parametrize("package", ["pytest", "ruff"])
def test_get_package_version(package: str) -> None:
    assert isinstance(get_package_version(package), Version)


@packaging_available
def test_get_package_version_missing() -> None:
    assert get_package_version("missing") is None


def test_get_package_version_missing_packaging() -> None:
    with (
        patch("coola.utils.imports.optional.is_packaging_available", lambda: False),
        pytest.raises(
            MissingPackageError, match=r"'packaging' package is required but not installed."
        ),
    ):
        get_package_version("my_package")


@packaging_available
def test_get_package_version_invalid_version() -> None:
    with patch("coola.utils.version.version", return_value="not-a-version"):
        assert get_package_version("invalid_version_package") is None


@packaging_available
def test_get_package_version_is_cached() -> None:
    with patch("coola.utils.version.version", return_value="1.0"):
        get_package_version("cached_package")
    with patch("coola.utils.version.version", return_value="2.0"):
        assert get_package_version("cached_package") == Version("1.0")
    clear_availability_caches()


@packaging_available
def test_get_package_version_cache_is_cleared() -> None:
    with patch("coola.utils.version.version", return_value="1.0"):
        get_package_version("cleared_package")
    clear_availability_caches()
    with patch("coola.utils.version.version", return_value="2.0"):
        assert get_package_version("cleared_package") == Version("2.0")
    clear_availability_caches()
