from __future__ import annotations

from typing import Any

import pytest

from coola.iterator.dfs import (
    BaseIterator,
    DefaultIterator,
    IterableIterator,
    IteratorRegistry,
    MappingIterator,
)
from tests.unit.iterator.dfs.helpers import DEFAULT_SAMPLES, CustomList

######################################
#     Tests for IteratorRegistry     #
######################################


def test_iterator_registry_init_empty() -> None:
    registry = IteratorRegistry()
    assert len(registry._registry) == 0


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


def test_iterator_registry_find_iterator_most_specific() -> None:
    base_iterator = IterableIterator()
    specific_iterator = MappingIterator()
    registry = IteratorRegistry({list: base_iterator, CustomList: specific_iterator})

    assert registry.find_iterator(CustomList) is specific_iterator


@pytest.mark.parametrize(("data", "expected"), DEFAULT_SAMPLES)
def test_iterator_registry_iterate_default(data: Any, expected: Any) -> None:
    assert list(IteratorRegistry({object: DefaultIterator()}).iterate(data)) == expected


def test_iterator_registry_registry_isolation() -> None:
    iterator1 = IterableIterator()
    iterator2 = MappingIterator()

    registry1 = IteratorRegistry({list: iterator1})
    registry2 = IteratorRegistry({list: iterator2})

    assert registry1._registry[list] is iterator1
    assert registry2._registry[list] is iterator2
