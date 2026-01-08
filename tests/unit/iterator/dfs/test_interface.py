from __future__ import annotations

from collections.abc import Generator, Iterable, Mapping
from typing import Any

import pytest

from coola import objects_are_equal
from coola.iterator import dfs_iterate
from coola.iterator.dfs import (
    DefaultIterator,
    IterableIterator,
    IteratorRegistry,
    MappingIterator,
    get_default_registry,
    register_iterators,
)
from tests.unit.iterator.dfs.helpers import DEFAULT_SAMPLES, SAMPLES, CustomList


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry
    yield
    if hasattr(get_default_registry, "_registry"):
        del get_default_registry._registry


#################################
#     Tests for dfs_iterate     #
#################################


@pytest.mark.parametrize(("data", "expected"), SAMPLES)
def test_dfs_iterate(data: Any, expected: Any) -> None:
    assert objects_are_equal(list(dfs_iterate(data)), expected)


@pytest.mark.parametrize(("data", "expected"), DEFAULT_SAMPLES)
def test_dfs_iterate_custom_registry(data: Any, expected: Any) -> None:
    assert objects_are_equal(
        list(dfs_iterate(data, registry=IteratorRegistry({object: DefaultIterator()}))), expected
    )


########################################
#     Tests for register_iterators     #
########################################


def test_register_iterators_calls_registry() -> None:
    register_iterators({CustomList: IterableIterator()})
    assert get_default_registry().has_iterator(CustomList)


def test_register_iterators_with_exist_ok_true() -> None:
    register_iterators({CustomList: MappingIterator()}, exist_ok=False)
    register_iterators({CustomList: IterableIterator()}, exist_ok=True)


def test_register_iterators_with_exist_ok_false() -> None:
    register_iterators({CustomList: MappingIterator()}, exist_ok=False)
    with pytest.raises(RuntimeError, match="already registered"):
        register_iterators({CustomList: IterableIterator()}, exist_ok=False)


##########################################
#     Tests for get_default_registry     #
##########################################


def test_get_default_registry_returns_registry() -> None:
    """Test that get_default_registry returns a IteratorRegistry
    instance."""
    registry = get_default_registry()
    assert isinstance(registry, IteratorRegistry)


def test_get_default_registry_returns_singleton() -> None:
    """Test that get_default_registry returns the same instance on
    multiple calls."""
    registry1 = get_default_registry()
    registry2 = get_default_registry()
    assert registry1 is registry2


@pytest.mark.parametrize("data_type", [int, float, complex, bool, str, bytes])
def test_get_default_registry_scalar_types(data_type: type) -> None:
    """Test that scalar types are registered with DefaultIterator."""
    registry = get_default_registry()
    assert registry.has_iterator(data_type)
    assert isinstance(registry.find_iterator(data_type), DefaultIterator)


@pytest.mark.parametrize("data_type", [list, tuple, range, Iterable, set, frozenset])
def test_get_default_registry_iterables(data_type: type) -> None:
    """Test that iterable types are registered with IterableIterator."""
    registry = get_default_registry()
    assert registry.has_iterator(data_type)
    assert isinstance(registry.find_iterator(data_type), IterableIterator)


@pytest.mark.parametrize("data_type", [dict, Mapping])
def test_register_default_iterators_registers_mappings(data_type: type) -> None:
    """Test that mapping types are registered with MappingIterator."""
    registry = get_default_registry()
    assert registry.has_iterator(data_type)
    assert isinstance(registry.find_iterator(data_type), MappingIterator)


def test_register_default_iterators_registers_object() -> None:
    """Test that object type is registered as catch-all."""
    registry = get_default_registry()
    assert registry.has_iterator(object)
    assert isinstance(registry.find_iterator(object), DefaultIterator)


@pytest.mark.parametrize(("data", "expected"), SAMPLES)
def test_default_registry_can_iterate(data: Any, expected: Any) -> None:
    """Test the behavior of the default iterator registry."""
    assert objects_are_equal(list(get_default_registry().iterate(data)), expected)


def test_get_default_registry_singleton_persists_modifications() -> None:
    """Test that modifications to the registry persist across calls."""
    registry1 = get_default_registry()
    assert not registry1.has_iterator(CustomList)
    registry1.register(CustomList, IterableIterator())
    assert registry1.has_iterator(CustomList)

    # Get registry again
    registry2 = get_default_registry()
    assert registry1 is registry2
    assert registry2.has_iterator(CustomList)
