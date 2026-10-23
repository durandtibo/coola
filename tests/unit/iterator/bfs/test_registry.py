from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from coola.iterator.bfs import (
    BaseChildFinder,
    ChildFinderRegistry,
    DefaultChildFinder,
    IterableChildFinder,
    MappingChildFinder,
)
from coola.registry import AlreadyRegisteredError
from tests.unit.iterator.bfs.helpers import (
    ITERATE_SAMPLES,
    CustomList,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


#########################################
#     Tests for ChildFinderRegistry     #
#########################################


def test_child_finder_registry_init_empty() -> None:
    registry = ChildFinderRegistry()
    assert len(registry._state) == 0


def test_child_finder_registry_init_with_registry() -> None:
    registry = ChildFinderRegistry({list: IterableChildFinder()})
    assert list in registry._state


def test_child_finder_registry_init_with_registry_keeps_the_child_finder() -> None:
    child_finder = IterableChildFinder()
    registry = ChildFinderRegistry({list: child_finder})
    assert registry._state[list] is child_finder


def test_child_finder_registry_init_with_registry_copies_the_registry() -> None:
    child_finder = IterableChildFinder()
    initial_registry: dict[type, BaseChildFinder[Any]] = {list: child_finder}
    registry = ChildFinderRegistry(initial_registry)
    initial_registry[tuple] = child_finder
    assert tuple not in registry._state


def test_child_finder_registry_repr() -> None:
    assert repr(ChildFinderRegistry()).startswith("ChildFinderRegistry(")


def test_child_finder_registry_str() -> None:
    assert str(ChildFinderRegistry()).startswith("ChildFinderRegistry(")


def test_child_finder_registry_register_new_type_has_child_finder() -> None:
    registry = ChildFinderRegistry()
    child_finder = IterableChildFinder()
    registry.register(list, child_finder)
    assert registry.has_child_finder(list)


def test_child_finder_registry_register_new_type_keeps_the_child_finder() -> None:
    registry = ChildFinderRegistry()
    child_finder = IterableChildFinder()
    registry.register(list, child_finder)
    assert registry._state[list] is child_finder


def test_child_finder_registry_register_existing_type_without_exist_ok() -> None:
    registry = ChildFinderRegistry()
    child_finder1 = IterableChildFinder()
    child_finder2 = MappingChildFinder()
    registry.register(list, child_finder1)
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        registry.register(list, child_finder2, exist_ok=False)


def test_child_finder_registry_register_existing_type_with_exist_ok() -> None:
    registry = ChildFinderRegistry()
    child_finder1 = IterableChildFinder()
    child_finder2 = MappingChildFinder()

    registry.register(list, child_finder1)
    registry.register(list, child_finder2, exist_ok=True)

    assert registry._state[list] is child_finder2


def test_child_finder_registry_register_many_list() -> None:
    registry = ChildFinderRegistry()
    registry.register_many(
        {
            list: IterableChildFinder(),
            dict: MappingChildFinder(),
            set: IterableChildFinder(),
        }
    )
    assert registry.has_child_finder(list)


def test_child_finder_registry_register_many_dict() -> None:
    registry = ChildFinderRegistry()
    registry.register_many(
        {
            list: IterableChildFinder(),
            dict: MappingChildFinder(),
            set: IterableChildFinder(),
        }
    )
    assert registry.has_child_finder(dict)


def test_child_finder_registry_register_many_set() -> None:
    registry = ChildFinderRegistry()
    registry.register_many(
        {
            list: IterableChildFinder(),
            dict: MappingChildFinder(),
            set: IterableChildFinder(),
        }
    )
    assert registry.has_child_finder(set)


def test_child_finder_registry_register_many_with_existing_type() -> None:
    registry = ChildFinderRegistry({object: DefaultChildFinder(), list: IterableChildFinder()})
    child_finders = {list: MappingChildFinder(), dict: MappingChildFinder()}
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        registry.register_many(child_finders, exist_ok=False)


def test_child_finder_registry_register_many_with_exist_ok() -> None:
    registry = ChildFinderRegistry({object: DefaultChildFinder()})
    child_finder1 = IterableChildFinder()
    registry.register(list, child_finder1)

    child_finder2 = MappingChildFinder()
    child_finders = {list: child_finder2, dict: MappingChildFinder()}

    registry.register_many(child_finders, exist_ok=True)
    assert registry._state[list] is child_finder2


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


def test_child_finder_registry_find_children_direct_match() -> None:
    registry = ChildFinderRegistry({list: IterableChildFinder()})
    assert list(registry.find_children([1, 2, 3])) == [1, 2, 3]


def test_child_finder_registry_find_children_mro_lookup() -> None:
    registry = ChildFinderRegistry({list: IterableChildFinder()})
    assert list(registry.find_children(CustomList([1, 2, 3]))) == [1, 2, 3]


def test_child_finder_registry_find_children_default() -> None:
    registry = ChildFinderRegistry({object: DefaultChildFinder(), list: IterableChildFinder()})
    assert list(registry.find_children(42)) == []


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


def test_child_finder_registry_iterate_duck_typed_iterable_no_finder_registered() -> None:
    r"""An object implementing ``__iter__`` but absent from any registered
    type's MRO must not be silently dropped when no dedicated child finder
    is registered for it: it should be yielded as a leaf value, not expanded
    into an empty result."""

    class CustomIterable:
        def __iter__(self) -> Any:
            yield 1
            yield 2

    data = CustomIterable()
    registry = ChildFinderRegistry({object: DefaultChildFinder()})

    assert list(registry.iterate(data)) == [data]


def test_child_finder_registry_iterate_container_type_without_registered_finder() -> None:
    r"""A structurally container-like value (e.g. a list) must not be
    dropped when no finder besides the default is registered: it should be
    yielded as a leaf value since the registry cannot expand it."""
    registry = ChildFinderRegistry({object: DefaultChildFinder()})

    assert list(registry.iterate([1, 2, 3])) == [[1, 2, 3]]


class LeafButRegisteredAsContainer:
    pass


def test_child_finder_registry_iterate_agrees_with_find_child_finder() -> None:
    r"""Containership must be decided solely from the resolved child
    finder, so a type registered with a non-default finder is always
    expanded, even if it would not structurally look like a
    container."""
    child_finder = IterableChildFinder()
    registry = ChildFinderRegistry(
        {object: DefaultChildFinder(), LeafButRegisteredAsContainer: child_finder}
    )
    assert registry.find_child_finder(LeafButRegisteredAsContainer) is child_finder


def test_child_finder_registry_iterate_expands_the_registered_type() -> None:
    registry = ChildFinderRegistry(
        {object: DefaultChildFinder(), LeafButRegisteredAsContainer: IterableChildFinder()}
    )
    # No __iter__, so IterableChildFinder.find_children raises when actually expanded.
    with pytest.raises(TypeError):
        list(registry.iterate(LeafButRegisteredAsContainer()))


def test_child_finder_registry_registry_isolation_first_registry() -> None:
    child_finder1 = IterableChildFinder()
    child_finder2 = MappingChildFinder()

    registry1 = ChildFinderRegistry({list: child_finder1})
    ChildFinderRegistry({list: child_finder2})
    assert registry1._state[list] is child_finder1


def test_child_finder_registry_registry_isolation_second_registry() -> None:
    child_finder1 = IterableChildFinder()
    child_finder2 = MappingChildFinder()

    ChildFinderRegistry({list: child_finder1})
    registry2 = ChildFinderRegistry({list: child_finder2})
    assert registry2._state[list] is child_finder2


############################################
#     Tests for is_leaf and the cycles     #
############################################


class LeafChildFinder(BaseChildFinder[Any]):
    r"""A leaf child finder that does not inherit from
    ``DefaultChildFinder``."""

    @property
    def is_leaf(self) -> bool:
        return True

    def find_children(self, data: Any) -> Iterator[Any]:  # noqa: ARG002
        return iter(())


class ExpandedDefaultChildFinder(DefaultChildFinder):
    r"""A child finder that inherits from ``DefaultChildFinder`` but has
    children."""

    @property
    def is_leaf(self) -> bool:
        return False

    def find_children(self, data: Any) -> Iterator[Any]:
        yield from data


class Wrapper:
    r"""A custom container."""

    def __init__(self, *items: Any) -> None:
        self.items = list(items)


class WrapperChildFinder(BaseChildFinder[Wrapper]):
    def find_children(self, data: Wrapper) -> Iterator[Any]:
        yield from data.items


def test_child_finder_is_leaf_default_is_false() -> None:
    assert not WrapperChildFinder().is_leaf


def test_default_child_finder_is_leaf() -> None:
    assert DefaultChildFinder().is_leaf


@pytest.mark.parametrize("child_finder", [IterableChildFinder(), MappingChildFinder()])
def test_container_child_finders_are_not_leaves(child_finder: BaseChildFinder[Any]) -> None:
    assert not child_finder.is_leaf


def test_child_finder_registry_iterate_custom_leaf_finder_not_inheriting_default() -> None:
    registry = ChildFinderRegistry(
        {object: DefaultChildFinder(), list: IterableChildFinder(), tuple: LeafChildFinder()}
    )
    assert list(registry.iterate([1, (2, 3), 4])) == [1, (2, 3), 4]


def test_child_finder_registry_iterate_default_subclass_with_children_is_expanded() -> None:
    registry = ChildFinderRegistry(
        {
            object: DefaultChildFinder(),
            list: IterableChildFinder(),
            tuple: ExpandedDefaultChildFinder(),
        }
    )
    assert list(registry.iterate([1, (2, 3)])) == [1, 2, 3]


def test_child_finder_registry_iterate_self_referencing_list_raises() -> None:
    data: list[Any] = [1]
    data.append(data)
    registry = ChildFinderRegistry({object: DefaultChildFinder(), list: IterableChildFinder()})
    with pytest.raises(ValueError, match="Cycle detected"):
        list(registry.iterate(data))


def test_child_finder_registry_iterate_self_referencing_dict_raises() -> None:
    data: dict[str, Any] = {"a": 1}
    data["self"] = data
    registry = ChildFinderRegistry({object: DefaultChildFinder(), dict: MappingChildFinder()})
    with pytest.raises(ValueError, match="Cycle detected"):
        list(registry.iterate(data))


def test_child_finder_registry_iterate_indirect_cycle_raises() -> None:
    first: list[Any] = []
    second: list[Any] = [first]
    first.append(second)
    registry = ChildFinderRegistry({object: DefaultChildFinder(), list: IterableChildFinder()})
    with pytest.raises(ValueError, match="Cycle detected"):
        list(registry.iterate(first))


def test_child_finder_registry_iterate_cycle_in_custom_container_raises() -> None:
    wrapper = Wrapper(1)
    wrapper.items.append(wrapper)
    registry = ChildFinderRegistry({object: DefaultChildFinder(), Wrapper: WrapperChildFinder()})
    with pytest.raises(ValueError, match="Cycle detected"):
        list(registry.iterate(wrapper))


def test_child_finder_registry_iterate_shared_container_is_not_a_cycle() -> None:
    shared = [1, 2]
    registry = ChildFinderRegistry({object: DefaultChildFinder(), list: IterableChildFinder()})
    assert list(registry.iterate([shared, shared, [shared]])) == [1, 2, 1, 2, 1, 2]


def test_child_finder_registry_iterate_repeated_leaf_is_not_a_cycle() -> None:
    value = "x"
    registry = ChildFinderRegistry({object: DefaultChildFinder(), list: IterableChildFinder()})
    assert list(registry.iterate([value, [value, value]])) == ["x", "x", "x"]


def test_child_finder_registry_iterate_not_leaf_without_children_is_not_yielded() -> None:
    class NoChildFinder(BaseChildFinder[Any]):
        def find_children(self, data: Any) -> Iterator[Any]:  # noqa: ARG002
            return iter(())

    registry = ChildFinderRegistry(
        {object: DefaultChildFinder(), list: IterableChildFinder(), complex: NoChildFinder()}
    )
    assert list(registry.iterate([1, 2j, 3])) == [1, 3]
