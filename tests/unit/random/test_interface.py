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
    """Reset the registry before and after each test."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry


class CustomList(list):
    r"""Create a custom class that inherits from list."""


#######################################
#     Tests for register_managers     #
#######################################


def test_register_managers_calls_registry() -> None:
    register_managers({"custom": RandomRandomManager()})
    assert get_default_registry().has_manager("custom")


def test_register_managers_with_exist_ok_true() -> None:
    register_managers({"custom": RandomRandomManager()})
    register_managers({"custom": RandomRandomManager()}, exist_ok=True)


def test_register_managers_with_exist_ok_false() -> None:
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
    """Test that get_default_registry returns the same instance on
    multiple calls."""
    registry1 = get_default_registry()
    registry2 = get_default_registry()
    assert registry1 is registry2


def test_get_default_registry_default() -> None:
    """Test that scalar types are registered with DefaultTransformer."""
    registry = get_default_registry()
    assert registry.has_manager("random")


@numpy_available
def test_get_default_registry_with_numpy() -> None:
    """Test that sequence types are registered with
    SequenceTransformer."""
    registry = get_default_registry()
    assert registry.has_manager("numpy")


@numpy_not_available
def test_get_default_registry_without_numpy() -> None:
    """Test that sequence types are registered with
    SequenceTransformer."""
    registry = get_default_registry()
    assert not registry.has_manager("numpy")


@torch_available
def test_get_default_registry_with_torch() -> None:
    """Test that sequence types are registered with
    SequenceTransformer."""
    registry = get_default_registry()
    assert registry.has_manager("torch")


@torch_not_available
def test_get_default_registry_without_torch() -> None:
    """Test that sequence types are registered with
    SequenceTransformer."""
    registry = get_default_registry()
    assert not registry.has_manager("torch")


def test_get_default_registry_singleton_persists_modifications() -> None:
    """Test that modifications to the registry persist across calls."""
    registry1 = get_default_registry()
    assert not registry1.has_manager("custom")
    registry1.register("custom", RandomRandomManager())
    assert registry1.has_manager("custom")

    # Get registry again
    registry2 = get_default_registry()
    assert registry1 is registry2
    assert registry2.has_manager("custom")
