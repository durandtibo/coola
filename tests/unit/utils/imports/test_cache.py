from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from coola.utils.imports import (
    clear_availability_caches,
    is_numpy_available,
    module_available,
    package_available,
)

if TYPE_CHECKING:
    from collections.abc import Generator

MODULE = "coola.utils.imports.universal"
logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _cache_clear() -> Generator[None]:
    clear_availability_caches()
    yield
    clear_availability_caches()


def test_clear_availability_caches_forces_the_checks_to_run_again() -> None:
    def broken_import(name: str) -> None:
        msg = f"cannot import {name}"
        raise OSError(msg)

    first = is_numpy_available()
    with patch(f"{MODULE}.importlib.import_module", broken_import):
        assert is_numpy_available() == first  # cached: the broken import is not seen
        clear_availability_caches()
        assert not is_numpy_available()  # checked again: the module cannot be imported
    clear_availability_caches()
    assert is_numpy_available() == first


def test_clear_availability_caches_clears_module_available() -> None:
    assert module_available("json")
    with patch(f"{MODULE}.importlib.import_module", side_effect=ImportError):
        assert module_available("json")  # cached
        clear_availability_caches()
        assert not module_available("json")


def test_clear_availability_caches_clears_package_available() -> None:
    assert package_available("json")
    with patch(f"{MODULE}.find_spec", return_value=None):
        assert package_available("json")  # cached
        clear_availability_caches()
        assert not package_available("json")


def test_is_numpy_available_reports_broken_installation_as_unavailable() -> None:
    with patch(f"{MODULE}.importlib.import_module", side_effect=OSError("libblas.so missing")):
        assert not is_numpy_available()
