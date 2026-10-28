from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from coola.utils import imports
from coola.utils.imports import (
    availability_cache,
    clear_availability_caches,
    is_numpy_available,
    module_available,
    package_available,
)

if TYPE_CHECKING:
    from collections.abc import Generator

from coola.utils.imports.package import _CACHED_FUNCTIONS

MODULE = "coola.utils.imports.package"
logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> Generator[None]:
    clear_availability_caches()
    yield
    clear_availability_caches()


def broken_import(name: str) -> None:
    msg = f"cannot import {name}"
    raise OSError(msg)


def test_availability_is_cached_between_calls() -> None:
    first = is_numpy_available()
    with patch(f"{MODULE}.importlib.import_module", broken_import):
        assert is_numpy_available() == first  # cached: the broken import is not seen


def test_clear_availability_caches_forces_the_checks_to_run_again() -> None:
    is_numpy_available()
    with patch(f"{MODULE}.importlib.import_module", broken_import):
        clear_availability_caches()
        assert not is_numpy_available()  # checked again: the module cannot be imported
    clear_availability_caches()


def test_clear_availability_caches_restores_the_availability() -> None:
    first = is_numpy_available()
    with patch(f"{MODULE}.importlib.import_module", broken_import):
        clear_availability_caches()
        is_numpy_available()
    clear_availability_caches()
    assert is_numpy_available() == first


def test_module_available_is_cached() -> None:
    module_available("json")
    with patch(f"{MODULE}.importlib.import_module", side_effect=ImportError):
        assert module_available("json")  # cached


def test_clear_availability_caches_clears_module_available() -> None:
    module_available("json")
    with patch(f"{MODULE}.importlib.import_module", side_effect=ImportError):
        clear_availability_caches()
        assert not module_available("json")
    clear_availability_caches()


def test_package_available_is_cached() -> None:
    package_available("json")
    with patch(f"{MODULE}.find_spec", return_value=None):
        assert package_available("json")  # cached


def test_clear_availability_caches_clears_package_available() -> None:
    package_available("json")
    with patch(f"{MODULE}.find_spec", return_value=None):
        clear_availability_caches()
        assert not package_available("json")
    clear_availability_caches()


def test_is_numpy_available_reports_broken_installation_as_unavailable() -> None:
    with patch(f"{MODULE}.importlib.import_module", side_effect=OSError("libblas.so missing")):
        assert not is_numpy_available()


IS_AVAILABLE_NAMES = [
    name for name in imports.__all__ if name.startswith("is_") and name.endswith("_available")
]


def test_there_are_is_available_functions() -> None:
    assert len(IS_AVAILABLE_NAMES) >= 12


@pytest.mark.parametrize("name", IS_AVAILABLE_NAMES)
def test_all_is_available_functions_are_registered(name: str) -> None:
    # A new ``is_<package>_available`` that is not decorated with
    # ``availability_cache`` would not be reset by ``clear_availability_caches``.
    assert getattr(imports, name) in _CACHED_FUNCTIONS


def test_package_available_is_registered() -> None:
    assert package_available in _CACHED_FUNCTIONS


def test_module_available_is_registered() -> None:
    assert module_available in _CACHED_FUNCTIONS


def test_availability_cache_caches() -> None:
    calls = []

    @availability_cache
    def is_my_package_available() -> bool:
        calls.append(1)
        return True

    try:
        is_my_package_available()
        is_my_package_available()
        assert len(calls) == 1
    finally:
        _CACHED_FUNCTIONS.remove(is_my_package_available)


def test_availability_cache_returns_the_value() -> None:
    @availability_cache
    def is_my_package_available() -> bool:
        return True

    try:
        assert is_my_package_available()
    finally:
        _CACHED_FUNCTIONS.remove(is_my_package_available)


def test_availability_cache_registers() -> None:
    @availability_cache
    def is_my_package_available() -> bool:
        return True

    try:
        assert is_my_package_available in _CACHED_FUNCTIONS
    finally:
        _CACHED_FUNCTIONS.remove(is_my_package_available)


def test_availability_cache_is_cleared_by_clear_availability_caches() -> None:
    calls = []

    @availability_cache
    def is_my_package_available() -> bool:
        calls.append(1)
        return True

    try:
        is_my_package_available()
        clear_availability_caches()
        is_my_package_available()
        assert len(calls) == 2
    finally:
        _CACHED_FUNCTIONS.remove(is_my_package_available)
