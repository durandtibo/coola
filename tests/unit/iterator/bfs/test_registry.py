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
    ITERATE_SAMPLES,
    CustomList,
)

#########################################
#     Tests for ChildFinderRegistry     #
#########################################


def test_child_finder_registry_init_empty() -> None:
    registry = ChildFinderRegistry()
    assert len(registry._registry) == 0


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
    registry = ChildFinderRegistry({object: DefaultChildFinder(), list: IterableChildFinder()})
    child_finders = {list: MappingChildFinder(), dict: MappingChildFinder()}
    with pytest.raises(RuntimeError, match="already registered"):
        registry.register_many(child_finders, exist_ok=False)


def test_child_finder_registry_register_many_with_exist_ok() -> None:
    registry = ChildFinderRegistry({object: DefaultChildFinder()})
    child_finder1 = IterableChildFinder()
    registry.register(list, child_finder1)

    child_finder2 = MappingChildFinder()
    child_finders = {list: child_finder2, dict: MappingChildFinder()}

    registry.register_many(child_finders, exist_ok=True)
    assert registry._registry[list] is child_finder2


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


def test_child_finder_registry_find_child_finder_most_specific() -> None:
    base_child_finder = IterableChildFinder()
    specific_child_finder = MappingChildFinder()
    registry = ChildFinderRegistry({list: base_child_finder, CustomList: specific_child_finder})
    assert registry.find_child_finder(CustomList) is specific_child_finder


@pytest.mark.parametrize(("data", "expected"), ITERATE_SAMPLES)
def test_child_finder_registry_iterate(data: Any, expected: Any) -> None:
    iterable_child_finder = IterableChildFinder()
    mapping_child_finder = MappingChildFinder()
    assert (
        list(
            ChildFinderRegistry(
                {
                    object: DefaultChildFinder(),
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


def test_child_finder_registry_registry_isolation() -> None:
    child_finder1 = IterableChildFinder()
    child_finder2 = MappingChildFinder()

    registry1 = ChildFinderRegistry({list: child_finder1})
    registry2 = ChildFinderRegistry({list: child_finder2})

    assert registry1._registry[list] is child_finder1
    assert registry2._registry[list] is child_finder2
