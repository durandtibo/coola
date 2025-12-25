from __future__ import annotations

from collections.abc import Generator, Iterable, Mapping
from typing import Any

import pytest

from coola import objects_are_equal
from coola.iterator import bfs_iterate
from coola.iterator.bfs import (
    ChildFinderRegistry,
    IterableChildFinder,
    MappingChildFinder,
    get_default_registry,
    register_child_finders,
)
from tests.unit.iterator.bfs.helpers import (
    DEFAULT_ITERATE_SAMPLES,
    ITERATE_SAMPLES,
    CustomList,
)


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry


#################################
#     Tests for bfs_iterate     #
#################################


@pytest.mark.parametrize(("data", "expected"), ITERATE_SAMPLES)
def test_bfs_iterate(data: Any, expected: Any) -> None:
    assert objects_are_equal(list(bfs_iterate(data)), expected)


@pytest.mark.parametrize(("data", "expected"), DEFAULT_ITERATE_SAMPLES)
def test_bfs_iterate_custom_registry(data: Any, expected: Any) -> None:
    assert objects_are_equal(list(bfs_iterate(data, registry=ChildFinderRegistry())), expected)


############################################
#     Tests for register_child_finders     #
############################################


def test_register_child_finders_calls_registry() -> None:
    register_child_finders({CustomList: IterableChildFinder()})
    assert get_default_registry().has_child_finder(CustomList)


def test_register_child_finders_with_exist_ok_true() -> None:
    register_child_finders({CustomList: MappingChildFinder()}, exist_ok=False)
    register_child_finders({CustomList: IterableChildFinder()}, exist_ok=True)


def test_register_child_finders_with_exist_ok_false() -> None:
    register_child_finders({CustomList: MappingChildFinder()}, exist_ok=False)
    with pytest.raises(RuntimeError, match="already registered"):
        register_child_finders({CustomList: IterableChildFinder()}, exist_ok=False)


##########################################
#     Tests for get_default_registry     #
##########################################


def test_get_default_registry_returns_registry() -> None:
    """Test that get_default_registry returns a ChildFinderRegistry
    instance."""
    registry = get_default_registry()
    assert isinstance(registry, ChildFinderRegistry)


def test_get_default_registry_returns_singleton() -> None:
    """Test that get_default_registry returns the same instance on
    multiple calls."""
    registry1 = get_default_registry()
    registry2 = get_default_registry()
    assert registry1 is registry2


def test_get_default_registry_scalar_types() -> None:
    """Test that scalar types are registered with DefaultChildFinder."""
    registry = get_default_registry()
    assert registry.has_child_finder(int)
    assert registry.has_child_finder(float)
    assert registry.has_child_finder(complex)
    assert registry.has_child_finder(bool)
    assert registry.has_child_finder(str)


def test_get_default_registry_sequences() -> None:
    """Test that sequence types are registered with
    IterableChildFinder."""
    registry = get_default_registry()
    assert registry.has_child_finder(list)
    assert registry.has_child_finder(tuple)
    assert registry.has_child_finder(range)
    assert registry.has_child_finder(Iterable)


def test_get_default_registry_sets() -> None:
    """Test that set types are registered with SetChildFinder."""
    registry = get_default_registry()
    assert registry.has_child_finder(set)
    assert registry.has_child_finder(frozenset)


def test_register_default_child_finders_registers_mappings() -> None:
    """Test that mapping types are registered with
    MappingChildFinder."""
    registry = get_default_registry()
    assert registry.has_child_finder(dict)
    assert registry.has_child_finder(Mapping)


def test_register_default_child_finders_registers_object() -> None:
    """Test that object type is registered as catch-all."""
    registry = get_default_registry()
    assert registry.has_child_finder(object)


@pytest.mark.parametrize(("data", "expected"), ITERATE_SAMPLES)
def test_default_registry_can_iterate(data: Any, expected: Any) -> None:
    """Test the behavior of the default child_finder registry."""
    assert objects_are_equal(list(get_default_registry().iterate(data)), expected)


def test_get_default_registry_singleton_persists_modifications() -> None:
    """Test that modifications to the registry persist across calls."""
    registry1 = get_default_registry()
    assert not registry1.has_child_finder(CustomList)
    registry1.register(CustomList, IterableChildFinder())
    assert registry1.has_child_finder(CustomList)

    # Get registry again
    registry2 = get_default_registry()
    assert registry1 is registry2
    assert registry2.has_child_finder(CustomList)
