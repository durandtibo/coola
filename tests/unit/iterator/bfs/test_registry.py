from __future__ import annotations

from typing import Any

import pytest

from coola.iterator.bfs import (
    BaseChildFinder,
    ChildFinderRegistry,
    DefaultChildFinder,
    IterableChildFinder,
    MappingChildFinder,
)
from tests.unit.iterator.bfs.helpers import (
    DEFAULT_FIND_CHILDREN_SAMPLES,
    DEFAULT_ITERATE_SAMPLES,
    ITERATE_SAMPLES,
    CustomList,
)

#########################################
#     Tests for ChildFinderRegistry     #
#########################################


def test_child_finder_registry_init_empty() -> None:
    registry = ChildFinderRegistry()
    assert registry._registry == {}
    assert isinstance(registry._default_child_finder, DefaultChildFinder)


def test_child_finder_registry_init_with_registry() -> None:
    child_finder = IterableChildFinder()
    initial_registry: dict[type, BaseChildFinder[Any]] = {list: child_finder}
    registry = ChildFinderRegistry(initial_registry)

    assert list in registry._registry
    assert registry._registry[list] is child_finder
    # Verify it's a copy
    initial_registry[tuple] = child_finder
    assert tuple not in registry._registry


def test_child_finder_registry_repr() -> None:
    assert repr(ChildFinderRegistry()).startswith("ChildFinderRegistry(")


def test_child_finder_registry_str() -> None:
    assert str(ChildFinderRegistry()).startswith("ChildFinderRegistry(")


def test_child_finder_registry_register_new_type() -> None:
    registry = ChildFinderRegistry()
    child_finder = IterableChildFinder()
    registry.register(list, child_finder)
    assert registry.has_child_finder(list)
    assert registry._registry[list] is child_finder


def test_child_finder_registry_register_existing_type_without_exist_ok() -> None:
    registry = ChildFinderRegistry()
    child_finder1 = IterableChildFinder()
    child_finder2 = MappingChildFinder()
    registry.register(list, child_finder1)
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register(list, child_finder2, exist_ok=False)


def test_child_finder_registry_register_existing_type_with_exist_ok() -> None:
    registry = ChildFinderRegistry()
    child_finder1 = IterableChildFinder()
    child_finder2 = MappingChildFinder()

    registry.register(list, child_finder1)
    registry.register(list, child_finder2, exist_ok=True)

    assert registry._registry[list] is child_finder2


def test_child_finder_registry_register_many() -> None:
    registry = ChildFinderRegistry()
    registry.register_many(
        {
            list: IterableChildFinder(),
            dict: MappingChildFinder(),
            set: IterableChildFinder(),
        }
    )
    assert registry.has_child_finder(list)
    assert registry.has_child_finder(dict)
    assert registry.has_child_finder(set)


def test_child_finder_registry_register_many_with_existing_type() -> None:
    registry = ChildFinderRegistry()
    registry.register(list, IterableChildFinder())
    child_finders = {list: MappingChildFinder(), dict: MappingChildFinder()}
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register_many(child_finders, exist_ok=False)


def test_child_finder_registry_register_many_with_exist_ok() -> None:
    registry = ChildFinderRegistry()
    child_finder1 = IterableChildFinder()
    registry.register(list, child_finder1)

    child_finder2 = MappingChildFinder()
    child_finders = {list: child_finder2, dict: MappingChildFinder()}

    registry.register_many(child_finders, exist_ok=True)
    assert registry._registry[list] is child_finder2


def test_child_finder_registry_register_clears_cache() -> None:
    """Test that registering a new child_finder clears the cache."""
    registry = ChildFinderRegistry()
    # Access find_child_finder to potentially populate cache
    assert isinstance(registry.find_child_finder(list), DefaultChildFinder)
    # Register should clear cache
    child_finder = IterableChildFinder()
    registry.register(list, child_finder)
    # Verify the new child_finder is found
    assert registry.find_child_finder(list) is child_finder


def test_child_finder_registry_has_child_finder_true() -> None:
    assert ChildFinderRegistry({list: IterableChildFinder()}).has_child_finder(list)


def test_child_finder_registry_has_child_finder_false() -> None:
    assert not ChildFinderRegistry().has_child_finder(list)


def test_child_finder_registry_find_child_finder_direct_match() -> None:
    child_finder = IterableChildFinder()
    registry = ChildFinderRegistry({list: child_finder})
    assert registry.find_child_finder(list) is child_finder


def test_child_finder_registry_find_child_finder_mro_lookup() -> None:
    child_finder = IterableChildFinder()
    registry = ChildFinderRegistry({list: child_finder})
    assert registry.find_child_finder(CustomList) is child_finder


def test_child_finder_registry_find_child_finder_default() -> None:
    assert isinstance(ChildFinderRegistry().find_child_finder(str), DefaultChildFinder)


def test_child_finder_registry_find_child_finder_most_specific() -> None:
    base_child_finder = IterableChildFinder()
    specific_child_finder = MappingChildFinder()
    registry = ChildFinderRegistry({list: base_child_finder, CustomList: specific_child_finder})
    assert registry.find_child_finder(CustomList) is specific_child_finder


def test_child_finder_registry_find_child_finder_cache() -> None:
    child_finder = IterableChildFinder()
    registry = ChildFinderRegistry({list: child_finder})

    assert registry.find_child_finder(CustomList) is child_finder
    assert registry._child_finder_cache == {CustomList: child_finder}

    assert registry.find_child_finder(CustomList) is child_finder
    assert registry._child_finder_cache == {CustomList: child_finder}

    assert registry.find_child_finder(list) is child_finder
    assert registry._child_finder_cache == {CustomList: child_finder, list: child_finder}


@pytest.mark.parametrize(("data", "expected"), DEFAULT_FIND_CHILDREN_SAMPLES)
def test_child_finder_registry_find_children_default(data: Any, expected: Any) -> None:
    assert list(ChildFinderRegistry().find_children(data)) == expected


@pytest.mark.parametrize(("data", "expected"), ITERATE_SAMPLES)
def test_child_finder_registry_iterate(data: Any, expected: Any) -> None:
    iterable_child_finder = IterableChildFinder()
    mapping_child_finder = MappingChildFinder()
    assert (
        list(
            ChildFinderRegistry(
                {
                    dict: mapping_child_finder,
                    list: iterable_child_finder,
                    range: iterable_child_finder,
                    set: iterable_child_finder,
                    tuple: iterable_child_finder,
                }
            ).iterate(data)
        )
        == expected
    )


@pytest.mark.parametrize(("data", "expected"), DEFAULT_ITERATE_SAMPLES)
def test_child_finder_registry_iterate_default(data: Any, expected: Any) -> None:
    assert list(ChildFinderRegistry().iterate(data)) == expected


def test_child_finder_registry_registry_isolation() -> None:
    child_finder1 = IterableChildFinder()
    child_finder2 = MappingChildFinder()

    registry1 = ChildFinderRegistry({list: child_finder1})
    registry2 = ChildFinderRegistry({list: child_finder2})

    assert registry1._registry[list] is child_finder1
    assert registry2._registry[list] is child_finder2
