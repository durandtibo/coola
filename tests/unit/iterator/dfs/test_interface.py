from __future__ import annotations

from collections import OrderedDict
from collections.abc import Generator, Iterable, Mapping
from typing import Any

import pytest

from coola import objects_are_equal
from coola.iterator.dfs import (
    IterableIterator,
    IteratorRegistry,
    MappingIterator,
    dfs_iterate,
    get_default_registry,
    register_iterators,
)
from tests.unit.iterator.dfs.test_default import DEFAULT_SAMPLES

SAMPLES = [
    pytest.param("abc", ["abc"], id="str"),
    pytest.param(42, [42], id="int"),
    pytest.param("", [""], id="empty string"),
    # iterable
    pytest.param([5, 3, 8, 1, 9, 2], [5, 3, 8, 1, 9, 2], id="list"),
    pytest.param([], [], id="empty list"),
    pytest.param((1, 2, 3), [1, 2, 3], id="tuple"),
    pytest.param((), [], id="empty tuple"),
    pytest.param([[1, 2, 3], [4, 5, 6]], [1, 2, 3, 4, 5, 6], id="nested list"),
    pytest.param(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [1, 2, 3, 4, 5, 6, 7, 8], id="deeply nested"
    ),
    pytest.param([1, "a", 2.5, None, True], [1, "a", 2.5, None, True], id="mixed types"),
    pytest.param(range(3), [0, 1, 2], id="generator"),
    # mapping
    pytest.param({"a": 1, "b": 2}, [1, 2], id="dict"),
    pytest.param({"a": {"b": 1, "c": 2}, "d": 3}, [1, 2, 3], id="nested dict"),
    pytest.param({}, [], id="empty dict"),
    pytest.param({"a": {}}, [], id="empty nested dict"),
    pytest.param({"x": [1, 2], "y": [3, 4]}, [1, 2, 3, 4], id="nested dict list"),
    pytest.param({"a": {"b": [1, 2], "c": 3}, "d": 4}, [1, 2, 3, 4], id="nested dict mixed types"),
    pytest.param(OrderedDict({"a": 1, "b": 2}), [1, 2], id="ordered dict"),
]


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


#################################
#     Tests for dfs_iterate     #
#################################


@pytest.mark.parametrize(("data", "expected"), SAMPLES)
def test_dfs_iterate(data: Any, expected: Any) -> None:
    assert objects_are_equal(list(dfs_iterate(data)), expected)


@pytest.mark.parametrize(("data", "expected"), DEFAULT_SAMPLES)
def test_dfs_iterate_custom_registry(data: Any, expected: Any) -> None:
    assert objects_are_equal(list(dfs_iterate(data, registry=IteratorRegistry())), expected)


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


def test_get_default_registry_scalar_types() -> None:
    """Test that scalar types are registered with DefaultIterator."""
    registry = get_default_registry()
    assert registry.has_iterator(int)
    assert registry.has_iterator(float)
    assert registry.has_iterator(complex)
    assert registry.has_iterator(bool)
    assert registry.has_iterator(str)


def test_get_default_registry_sequences() -> None:
    """Test that sequence types are registered with IterableIterator."""
    registry = get_default_registry()
    assert registry.has_iterator(list)
    assert registry.has_iterator(tuple)
    assert registry.has_iterator(range)
    assert registry.has_iterator(Iterable)


def test_get_default_registry_sets() -> None:
    """Test that set types are registered with SetIterator."""
    registry = get_default_registry()
    assert registry.has_iterator(set)
    assert registry.has_iterator(frozenset)


def test_register_default_iterators_registers_mappings() -> None:
    """Test that mapping types are registered with MappingIterator."""
    registry = get_default_registry()
    assert registry.has_iterator(dict)
    assert registry.has_iterator(Mapping)


def test_register_default_iterators_registers_object() -> None:
    """Test that object type is registered as catch-all."""
    registry = get_default_registry()
    assert registry.has_iterator(object)


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
