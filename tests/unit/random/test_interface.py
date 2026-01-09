from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from coola.random import (
    RandomManagerRegistry,
    RandomRandomManager,
    get_default_registry,
    register_managers,
)
from coola.testing import (
    numpy_available,
    numpy_not_available,
    torch_available,
    torch_not_available,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the default registry before and after each test to ensure
    test isolation."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry


class CustomList(list):
    r"""Custom list subclass for testing purposes."""


#######################################
#     Tests for register_managers     #
#######################################


def test_register_managers_calls_registry() -> None:
    """Test that register_managers successfully registers managers in
    the default registry."""
    register_managers({"custom": RandomRandomManager()})
    assert get_default_registry().has_manager("custom")


def test_register_managers_with_exist_ok_true() -> None:
    """Test that register_managers allows re-registration when
    exist_ok=True."""
    register_managers({"custom": RandomRandomManager()})
    register_managers({"custom": RandomRandomManager()}, exist_ok=True)


def test_register_managers_with_exist_ok_false() -> None:
    """Test that register_managers raises RuntimeError when attempting
    to re-register with exist_ok=False."""
    register_managers({"custom": RandomRandomManager()})
    with pytest.raises(RuntimeError, match="already registered"):
        register_managers({"custom": RandomRandomManager()}, exist_ok=False)


##########################################
#     Tests for get_default_registry     #
##########################################


def test_get_default_registry_returns_registry() -> None:
    """Test that get_default_registry returns a RandomManagerRegistry
    instance."""
    registry = get_default_registry()
    assert isinstance(registry, RandomManagerRegistry)


def test_get_default_registry_returns_singleton() -> None:
    """Test that get_default_registry returns the same singleton
    instance on multiple calls."""
    registry1 = get_default_registry()
    registry2 = get_default_registry()
    assert registry1 is registry2


def test_get_default_registry_default() -> None:
    """Test that the 'random' manager is registered by default."""
    registry = get_default_registry()
    assert registry.has_manager("random")


@numpy_available
def test_get_default_registry_with_numpy() -> None:
    """Test that the 'numpy' manager is registered when NumPy is
    available."""
    registry = get_default_registry()
    assert registry.has_manager("numpy")


@numpy_not_available
def test_get_default_registry_without_numpy() -> None:
    """Test that the 'numpy' manager is not registered when NumPy is
    unavailable."""
    registry = get_default_registry()
    assert not registry.has_manager("numpy")


@torch_available
def test_get_default_registry_with_torch() -> None:
    """Test that the 'torch' manager is registered when PyTorch is
    available."""
    registry = get_default_registry()
    assert registry.has_manager("torch")


@torch_not_available
def test_get_default_registry_without_torch() -> None:
    """Test that the 'torch' manager is not registered when PyTorch is
    unavailable."""
    registry = get_default_registry()
    assert not registry.has_manager("torch")


def test_get_default_registry_singleton_persists_modifications() -> None:
    """Test that modifications to the singleton registry persist across
    multiple calls to get_default_registry."""
    registry1 = get_default_registry()
    assert not registry1.has_manager("custom")
    registry1.register("custom", RandomRandomManager())
    assert registry1.has_manager("custom")

    # Get registry again and verify the modification persists
    registry2 = get_default_registry()
    assert registry1 is registry2
    assert registry2.has_manager("custom")
