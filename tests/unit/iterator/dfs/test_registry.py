from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pytest

from coola import objects_are_equal
from coola.iterator.dfs import (
    BaseIterator,
    DefaultIterator,
    IterableIterator,
    IteratorRegistry,
    MappingIterator,
)


class CustomList(list):
    r"""Create a custom class that inherits from list."""


######################################
#     Tests for IteratorRegistry     #
######################################


def test_iterator_registry_init_empty() -> None:
    registry = IteratorRegistry()
    assert registry._registry == {}
    assert isinstance(registry._default_iterator, DefaultIterator)


def test_iterator_registry_init_with_registry() -> None:
    iterator = IterableIterator()
    initial_registry: dict[type, BaseIterator[Any]] = {list: iterator}
    registry = IteratorRegistry(initial_registry)

    assert list in registry._registry
    assert registry._registry[list] is iterator
    # Verify it's a copy
    initial_registry[tuple] = iterator
    assert tuple not in registry._registry


def test_iterator_registry_repr() -> None:
    assert repr(IteratorRegistry()).startswith("IteratorRegistry(")


def test_iterator_registry_str() -> None:
    assert str(IteratorRegistry()).startswith("IteratorRegistry(")


def test_iterator_registry_register_new_type() -> None:
    registry = IteratorRegistry()
    iterator = IterableIterator()
    registry.register(list, iterator)
    assert registry.has_iterator(list)
    assert registry._registry[list] is iterator


def test_iterator_registry_register_existing_type_without_exist_ok() -> None:
    registry = IteratorRegistry()
    iterator1 = IterableIterator()
    iterator2 = MappingIterator()
    registry.register(list, iterator1)
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register(list, iterator2, exist_ok=False)


def test_iterator_registry_register_existing_type_with_exist_ok() -> None:
    registry = IteratorRegistry()
    iterator1 = IterableIterator()
    iterator2 = MappingIterator()

    registry.register(list, iterator1)
    registry.register(list, iterator2, exist_ok=True)

    assert registry._registry[list] is iterator2


def test_iterator_registry_register_many() -> None:
    registry = IteratorRegistry()
    registry.register_many(
        {
            list: IterableIterator(),
            dict: MappingIterator(),
            set: IterableIterator(),
        }
    )
    assert registry.has_iterator(list)
    assert registry.has_iterator(dict)
    assert registry.has_iterator(set)


def test_iterator_registry_register_many_with_existing_type() -> None:
    registry = IteratorRegistry()
    registry.register(list, IterableIterator())
    iterators = {list: MappingIterator(), dict: MappingIterator()}
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register_many(iterators, exist_ok=False)


def test_iterator_registry_register_many_with_exist_ok() -> None:
    registry = IteratorRegistry()
    iterator1 = IterableIterator()
    registry.register(list, iterator1)

    iterator2 = MappingIterator()
    iterators = {list: iterator2, dict: MappingIterator()}

    registry.register_many(iterators, exist_ok=True)
    assert registry._registry[list] is iterator2


def test_iterator_registry_register_clears_cache() -> None:
    """Test that registering a new iterator clears the cache."""
    registry = IteratorRegistry()
    # Access find_iterator to potentially populate cache
    assert isinstance(registry.find_iterator(list), DefaultIterator)
    # Register should clear cache
    iterator = IterableIterator()
    registry.register(list, iterator)
    # Verify the new iterator is found
    assert registry.find_iterator(list) is iterator


def test_iterator_registry_has_iterator_true() -> None:
    assert IteratorRegistry({list: IterableIterator()}).has_iterator(list)


def test_iterator_registry_has_iterator_false() -> None:
    assert not IteratorRegistry().has_iterator(list)


def test_iterator_registry_find_iterator_direct_match() -> None:
    iterator = IterableIterator()
    registry = IteratorRegistry({list: iterator})
    assert registry.find_iterator(list) is iterator


def test_iterator_registry_find_iterator_mro_lookup() -> None:
    iterator = IterableIterator()
    registry = IteratorRegistry({list: iterator})
    assert registry.find_iterator(CustomList) is iterator


def test_iterator_registry_find_iterator_default() -> None:
    assert isinstance(IteratorRegistry().find_iterator(str), DefaultIterator)


def test_iterator_registry_find_iterator_most_specific() -> None:
    base_iterator = IterableIterator()
    specific_iterator = MappingIterator()
    registry = IteratorRegistry({list: base_iterator, CustomList: specific_iterator})

    assert registry.find_iterator(CustomList) is specific_iterator


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param("abc", ["abc"], id="str"),
        pytest.param(42, [42], id="int"),
        pytest.param({"key": "value"}, [{"key": "value"}], id="dict"),
        pytest.param([1, 2, 3], [[1, 2, 3]], id="list"),
        pytest.param((1, 2, 3, 4), [(1, 2, 3, 4)], id="tuple"),
        pytest.param({1, 2, 3}, [{1, 2, 3}], id="set"),
    ],
)
def test_iterator_registry_iterate_default(data: Any, expected: Any) -> None:
    assert list(IteratorRegistry().iterate(data)) == expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [
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
        pytest.param(
            {"a": {"b": [1, 2], "c": 3}, "d": 4}, [1, 2, 3, 4], id="nested dict mixed types"
        ),
        pytest.param(OrderedDict({"a": 1, "b": 2}), [1, 2], id="ordered dict"),
    ],
)
def test_iterable_iterator_iterate(data: Any, expected: Any) -> None:
    assert objects_are_equal(list(IteratorRegistry().iterate(data)), expected)


def test_iterator_registry_registry_isolation() -> None:
    iterator1 = IterableIterator()
    iterator2 = MappingIterator()

    registry1 = IteratorRegistry({list: iterator1})
    registry2 = IteratorRegistry({list: iterator2})

    assert registry1._registry[list] is iterator1
    assert registry2._registry[list] is iterator2
