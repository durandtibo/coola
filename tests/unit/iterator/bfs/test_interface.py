from __future__ import annotations

from collections import UserString, deque
from collections.abc import Collection, Generator, Iterator, Mapping
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

import pytest

from coola.equality import objects_are_equal
from coola.iterator import bfs_iterate
from coola.iterator.bfs import (
    ChildFinderRegistry,
    DefaultChildFinder,
    IterableChildFinder,
    MappingChildFinder,
    get_default_registry,
    interface as bfs_interface,
    register_child_finders,
)
from coola.registry import AlreadyRegisteredError
from tests.unit.iterator.bfs.helpers import (
    DEFAULT_ITERATE_SAMPLES,
    ITERATE_SAMPLES,
    CustomList,
)


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    bfs_interface._default_registry.reset()
    yield
    bfs_interface._default_registry.reset()


#################################
#     Tests for bfs_iterate     #
#################################


@pytest.mark.parametrize(("data", "expected"), ITERATE_SAMPLES)
def test_bfs_iterate(data: Any, expected: Any) -> None:
    assert objects_are_equal(list(bfs_iterate(data)), expected)


@pytest.mark.parametrize(("data", "expected"), DEFAULT_ITERATE_SAMPLES)
def test_bfs_iterate_custom_registry(data: Any, expected: Any) -> None:
    assert objects_are_equal(
        list(bfs_iterate(data, registry=ChildFinderRegistry({object: DefaultChildFinder()}))),
        expected,
    )


def test_bfs_iterate_duck_typed_iterable_not_silently_dropped() -> None:
    r"""An object that implements ``__iter__`` without ``list`` (or any
    registered type) in its MRO must be yielded as a leaf value rather
    than silently dropped because the registry falls back to
    ``DefaultChildFinder``."""

    class CustomIterable:
        def __iter__(self) -> Any:
            yield 1
            yield 2

    data = CustomIterable()
    assert list(
        bfs_iterate(data, registry=ChildFinderRegistry({object: DefaultChildFinder()}))
    ) == [data]


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
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
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


@pytest.mark.parametrize("data_type", [int, float, complex, bool, str, bytes])
def test_get_default_registry_scalar_types_is_registered(data_type: type) -> None:
    """Test that scalar types are registered with DefaultChildFinder."""
    registry = get_default_registry()
    assert registry.has_child_finder(data_type)


@pytest.mark.parametrize("data_type", [int, float, complex, bool, str, bytes])
def test_get_default_registry_scalar_types_finder(data_type: type) -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_child_finder(data_type), DefaultChildFinder)


@pytest.mark.parametrize("data_type", [list, tuple, deque, range, Collection, set, frozenset])
def test_get_default_registry_iterable_is_registered(data_type: type) -> None:
    """Test that iterable types are registered with
    IterableChildFinder."""
    registry = get_default_registry()
    assert registry.has_child_finder(data_type)


@pytest.mark.parametrize("data_type", [list, tuple, deque, range, Collection, set, frozenset])
def test_get_default_registry_iterable_finder(data_type: type) -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_child_finder(data_type), IterableChildFinder)


@pytest.mark.parametrize("data_type", [dict, Mapping])
def test_register_default_child_finders_registers_mappings_is_registered(data_type: type) -> None:
    """Test that mapping types are registered with
    MappingChildFinder."""
    registry = get_default_registry()
    assert registry.has_child_finder(data_type)


@pytest.mark.parametrize("data_type", [dict, Mapping])
def test_register_default_child_finders_registers_mappings_finder(data_type: type) -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_child_finder(data_type), MappingChildFinder)


def test_register_default_child_finders_registers_object_is_registered() -> None:
    """Test that object type is registered as catch-all."""
    registry = get_default_registry()
    assert registry.has_child_finder(object)


def test_register_default_child_finders_registers_object_finder() -> None:
    registry = get_default_registry()
    assert isinstance(registry.find_child_finder(object), DefaultChildFinder)


@pytest.mark.parametrize(("data", "expected"), ITERATE_SAMPLES)
def test_default_registry_can_iterate(data: Any, expected: Any) -> None:
    """Test the behavior of the default child_finder registry."""
    assert objects_are_equal(list(get_default_registry().iterate(data)), expected)


def test_get_default_registry_singleton_does_not_have_the_custom_type() -> None:
    """Test that the registry does not have the custom type."""
    assert not get_default_registry().has_child_finder(CustomList)


def test_get_default_registry_singleton_persists_modifications() -> None:
    """Test that modifications to the registry persist."""
    registry1 = get_default_registry()
    registry1.register(CustomList, IterableChildFinder())
    assert registry1.has_child_finder(CustomList)


def test_get_default_registry_singleton_returns_the_same_registry() -> None:
    """Test that the registry is the same across calls."""
    registry1 = get_default_registry()
    registry1.register(CustomList, IterableChildFinder())
    assert registry1 is get_default_registry()


def test_get_default_registry_singleton_persists_modifications_across_calls() -> None:
    """Test that modifications to the registry persist across calls."""
    get_default_registry().register(CustomList, IterableChildFinder())
    assert get_default_registry().has_child_finder(CustomList)


class CustomIterator(Iterator):
    r"""An iterator that explicitly inherits from
    ``collections.abc.Iterator``."""

    def __init__(self, values: list[int]) -> None:
        self._values = iter(values)

    def __next__(self) -> int:
        return next(self._values)


def test_bfs_iterate_iterators_are_leaves_and_not_consumed_first_is_the_iterator() -> None:
    iterator = CustomIterator([1, 2, 3])
    generator = (i for i in range(3))
    result = list(bfs_iterate([iterator, generator, 4]))
    assert result[0] is iterator


def test_bfs_iterate_iterators_are_leaves_and_not_consumed_second_is_the_generator() -> None:
    iterator = CustomIterator([1, 2, 3])
    generator = (i for i in range(3))
    result = list(bfs_iterate([iterator, generator, 4]))
    assert result[1] is generator


def test_bfs_iterate_iterators_are_leaves_and_not_consumed_third_value() -> None:
    iterator = CustomIterator([1, 2, 3])
    generator = (i for i in range(3))
    result = list(bfs_iterate([iterator, generator, 4]))
    assert result[2] == 4


def test_bfs_iterate_iterators_are_leaves_and_not_consumed_iterator_is_not_consumed() -> None:
    iterator = CustomIterator([1, 2, 3])
    generator = (i for i in range(3))
    list(bfs_iterate([iterator, generator, 4]))
    assert list(iterator) == [1, 2, 3]


def test_bfs_iterate_iterators_are_leaves_and_not_consumed_generator_is_not_consumed() -> None:
    iterator = CustomIterator([1, 2, 3])
    generator = (i for i in range(3))
    list(bfs_iterate([iterator, generator, 4]))
    assert list(generator) == [0, 1, 2]


def test_bfs_iterate_self_referencing_list_raises() -> None:
    data: list = [1]
    data.append(data)
    with pytest.raises(ValueError, match="Cycle detected"):
        list(bfs_iterate(data))


def test_default_registry_iterates_over_a_deque() -> None:
    assert list(bfs_iterate(deque([1, [2, 3]]))) == [1, 2, 3]


def test_bfs_iterate_mapping_proxy() -> None:
    assert list(bfs_iterate(MappingProxyType({"a": 1, "b": [2, 3]}))) == [1, 2, 3]


def test_bfs_iterate_mutual_cycle_raises() -> None:
    first: list = [1]
    second: list = [first]
    first.append(second)
    with pytest.raises(ValueError, match="Cycle detected"):
        list(bfs_iterate(first))


def _make_deep_chain() -> list:
    data: list = [1]
    for _ in range(50):
        data = [data]
    return data


def test_bfs_iterate_deep_chain_yields_the_leaf() -> None:
    assert list(bfs_iterate(_make_deep_chain())) == [1]


def test_bfs_iterate_deep_chain_does_not_walk_the_ancestors() -> None:
    with patch("coola.iterator.bfs.registry._is_ancestor") as mock_is_ancestor:
        list(bfs_iterate(_make_deep_chain()))
    mock_is_ancestor.assert_not_called()


def test_bfs_iterate_user_string_is_a_leaf() -> None:
    value = UserString("abc")
    assert list(bfs_iterate([value, 1])) == [value, 1]
