from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import pytest

from coola.iterator.dfs import (
    BaseIterator,
    DefaultIterator,
    IterableIterator,
    IteratorRegistry,
    MappingIterator,
    PathRegistry,
)
from coola.registry import AlreadyRegisteredError
from tests.unit.iterator.dfs.helpers import DEFAULT_SAMPLES, CustomList

if TYPE_CHECKING:
    from collections.abc import Iterator


######################################
#     Tests for IteratorRegistry     #
######################################


def test_iterator_registry_init_empty() -> None:
    registry = IteratorRegistry()
    assert len(registry._state) == 0


def test_iterator_registry_init_with_registry() -> None:
    registry = IteratorRegistry({list: IterableIterator()})
    assert list in registry._state


def test_iterator_registry_init_with_registry_keeps_the_iterator() -> None:
    iterator = IterableIterator()
    registry = IteratorRegistry({list: iterator})
    assert registry._state[list] is iterator


def test_iterator_registry_init_with_registry_copies_the_registry() -> None:
    iterator = IterableIterator()
    initial_registry: dict[type, BaseIterator[Any]] = {list: iterator}
    registry = IteratorRegistry(initial_registry)
    initial_registry[tuple] = iterator
    assert tuple not in registry._state


def test_iterator_registry_repr() -> None:
    assert repr(IteratorRegistry()).startswith("IteratorRegistry(")


def test_iterator_registry_str() -> None:
    assert str(IteratorRegistry()).startswith("IteratorRegistry(")


def test_iterator_registry_register_new_type_has_iterator() -> None:
    registry = IteratorRegistry()
    iterator = IterableIterator()
    registry.register(list, iterator)
    assert registry.has_iterator(list)


def test_iterator_registry_register_new_type_keeps_the_iterator() -> None:
    registry = IteratorRegistry()
    iterator = IterableIterator()
    registry.register(list, iterator)
    assert registry._state[list] is iterator


def test_iterator_registry_register_existing_type_without_exist_ok() -> None:
    registry = IteratorRegistry()
    iterator1 = IterableIterator()
    iterator2 = MappingIterator()
    registry.register(list, iterator1)
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        registry.register(list, iterator2, exist_ok=False)


def test_iterator_registry_register_existing_type_with_exist_ok() -> None:
    registry = IteratorRegistry()
    iterator1 = IterableIterator()
    iterator2 = MappingIterator()

    registry.register(list, iterator1)
    registry.register(list, iterator2, exist_ok=True)

    assert registry._state[list] is iterator2


def test_iterator_registry_register_many_list() -> None:
    registry = IteratorRegistry()
    registry.register_many(
        {
            list: IterableIterator(),
            dict: MappingIterator(),
            set: IterableIterator(),
        }
    )
    assert registry.has_iterator(list)


def test_iterator_registry_register_many_dict() -> None:
    registry = IteratorRegistry()
    registry.register_many(
        {
            list: IterableIterator(),
            dict: MappingIterator(),
            set: IterableIterator(),
        }
    )
    assert registry.has_iterator(dict)


def test_iterator_registry_register_many_set() -> None:
    registry = IteratorRegistry()
    registry.register_many(
        {
            list: IterableIterator(),
            dict: MappingIterator(),
            set: IterableIterator(),
        }
    )
    assert registry.has_iterator(set)


def test_iterator_registry_register_many_with_existing_type() -> None:
    registry = IteratorRegistry()
    registry.register(list, IterableIterator())
    iterators = {list: MappingIterator(), dict: MappingIterator()}
    with pytest.raises(AlreadyRegisteredError, match=r"already registered"):
        registry.register_many(iterators, exist_ok=False)


def test_iterator_registry_register_many_with_exist_ok() -> None:
    registry = IteratorRegistry()
    iterator1 = IterableIterator()
    registry.register(list, iterator1)

    iterator2 = MappingIterator()
    iterators = {list: iterator2, dict: MappingIterator()}

    registry.register_many(iterators, exist_ok=True)
    assert registry._state[list] is iterator2


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


def test_iterator_registry_registry_isolation_first_registry() -> None:
    iterator1 = IterableIterator()
    iterator2 = MappingIterator()

    registry1 = IteratorRegistry({list: iterator1})
    IteratorRegistry({list: iterator2})
    assert registry1._state[list] is iterator1


def test_iterator_registry_registry_isolation_second_registry() -> None:
    iterator1 = IterableIterator()
    iterator2 = MappingIterator()

    IteratorRegistry({list: iterator1})
    registry2 = IteratorRegistry({list: iterator2})
    assert registry2._state[list] is iterator2


################################
#     Tests for the cycles     #
################################


class Wrapper:
    r"""A custom container."""

    def __init__(self, *items: Any) -> None:
        self.items = list(items)


class WrapperIterator(BaseIterator[Wrapper]):
    def iterate(self, data: Wrapper, registry: IteratorRegistry) -> Iterator[Any]:
        for item in data.items:
            yield from registry.iterate(item)


def test_iterator_registry_iterate_self_referencing_list_raises() -> None:
    data: list[Any] = [1]
    data.append(data)
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    with pytest.raises(ValueError, match="Cycle detected"):
        list(registry.iterate(data))


def test_iterator_registry_iterate_self_referencing_dict_raises() -> None:
    data: dict[str, Any] = {"a": 1}
    data["self"] = data
    registry = IteratorRegistry({object: DefaultIterator(), dict: MappingIterator()})
    with pytest.raises(ValueError, match="Cycle detected"):
        list(registry.iterate(data))


def test_iterator_registry_iterate_indirect_cycle_raises() -> None:
    first: list[Any] = []
    second: list[Any] = [first]
    first.append(second)
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    with pytest.raises(ValueError, match="Cycle detected"):
        list(registry.iterate(first))


def test_iterator_registry_iterate_cycle_in_custom_container_raises() -> None:
    wrapper = Wrapper(1)
    wrapper.items.append(wrapper)
    registry = IteratorRegistry({object: DefaultIterator(), Wrapper: WrapperIterator()})
    with pytest.raises(ValueError, match="Cycle detected"):
        list(registry.iterate(wrapper))


def test_iterator_registry_iterate_shared_container_is_not_a_cycle() -> None:
    shared = [1, 2]
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    assert list(registry.iterate([shared, shared, [shared]])) == [1, 2, 1, 2, 1, 2]


def test_iterator_registry_iterate_repeated_leaf_is_not_a_cycle() -> None:
    value = "x"
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    assert list(registry.iterate([value, [value, value]])) == ["x", "x", "x"]


def test_iterator_registry_iterate_yields_the_first_item_before_the_registration() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    iterator = registry.iterate([1, Wrapper(2)])
    assert next(iterator) == 1


def test_iterator_registry_iterate_sees_registrations_made_after_a_traversal_started() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    iterator = registry.iterate([1, Wrapper(2)])
    next(iterator)
    registry.register(Wrapper, WrapperIterator())
    assert list(iterator) == [2]


##################################
#     Tests for PathRegistry     #
##################################


def test_path_registry_is_an_iterator_registry() -> None:
    assert isinstance(PathRegistry(IteratorRegistry(), ()), IteratorRegistry)


def test_path_registry_shares_the_state_of_the_registry() -> None:
    registry = IteratorRegistry({object: DefaultIterator()})
    path_registry = PathRegistry(registry, ())
    assert path_registry._state is registry._state


def test_path_registry_sees_the_registrations_of_the_registry() -> None:
    registry = IteratorRegistry({object: DefaultIterator()})
    path_registry = PathRegistry(registry, ())
    registry.register(list, IterableIterator())
    assert path_registry.has(list)


def test_path_registry_iterate() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    assert list(PathRegistry(registry, ()).iterate([1, [2, 3]])) == [1, 2, 3]


def test_path_registry_iterate_ancestor_raises() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    data = [1, 2]
    with pytest.raises(ValueError, match="Cycle detected"):
        list(PathRegistry(registry, (id(data),)).iterate(data))


def test_path_registry_iterate_not_an_ancestor() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    other = [1]  # keep a reference: the id of a freed object can be reused
    assert list(PathRegistry(registry, (id(other),)).iterate([1, 2])) == [1, 2]


class RecordingIterator(BaseIterator[Wrapper]):
    def __init__(self, received: list[IteratorRegistry], *, recursive: bool = False) -> None:
        self._received = received
        self._recursive = recursive

    def iterate(self, data: Wrapper, registry: IteratorRegistry) -> Iterator[Any]:
        self._received.append(registry)
        for item in data.items:
            if self._recursive:
                yield from registry.iterate(item)
            else:
                yield item


def test_path_registry_is_given_to_the_iterators_values() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), Wrapper: RecordingIterator([])})
    assert list(registry.iterate(Wrapper(1, 2))) == [1, 2]


def test_path_registry_is_given_to_the_iterators() -> None:
    received: list[IteratorRegistry] = []
    registry = IteratorRegistry({object: DefaultIterator(), Wrapper: RecordingIterator(received)})
    list(registry.iterate(Wrapper(1, 2)))
    assert isinstance(received[0], PathRegistry)


def _traverse_nested_wrappers() -> tuple[list[Any], list[IteratorRegistry]]:
    received: list[IteratorRegistry] = []
    registry = IteratorRegistry(
        {object: DefaultIterator(), Wrapper: RecordingIterator(received, recursive=True)}
    )
    data = Wrapper(Wrapper(1, 2), Wrapper(3, Wrapper(4)))
    return list(registry.iterate(data)), received


def test_path_registry_traverses_all_the_objects() -> None:
    values, _ = _traverse_nested_wrappers()
    assert values == [1, 2, 3, 4]


def test_path_registry_is_given_to_all_the_wrappers_of_a_traversal() -> None:
    _, received = _traverse_nested_wrappers()
    assert len(received) == 4


def test_path_registry_is_the_same_object_for_all_the_objects_of_a_traversal() -> None:
    _, received = _traverse_nested_wrappers()
    assert all(r is received[0] for r in received)


def test_path_registry_shared_container_is_not_a_cycle() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    path_registry = PathRegistry(registry)
    shared = [1]
    # ``shared`` is visited twice, but never inside itself: it is not a cycle.
    assert list(path_registry.iterate([shared, shared])) == [1, 1]


def test_path_registry_removes_the_ancestors_when_leaving_a_container() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    path_registry = PathRegistry(registry)
    shared = [1]
    list(path_registry.iterate([shared, shared]))
    assert path_registry._ancestors == set()


def _make_cycle() -> list:
    data: list = [1]
    data.append(data)
    return data


def test_path_registry_raises_an_error_for_a_cycle() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    path_registry = PathRegistry(registry)
    with pytest.raises(ValueError, match="Cycle detected"):
        list(path_registry.iterate(_make_cycle()))


def test_path_registry_removes_the_ancestors_after_an_error() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    path_registry = PathRegistry(registry)
    with contextlib.suppress(ValueError):
        list(path_registry.iterate(_make_cycle()))
    assert path_registry._ancestors == set()


def test_path_registry_keeps_the_ancestors_while_the_generator_is_running() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    path_registry = PathRegistry(registry)
    iterator = path_registry.iterate([[1, 2], [3, 4]])
    next(iterator)
    # The outer list, the first inner list and the leaf being yielded.
    assert len(path_registry._ancestors) == 3


def test_path_registry_removes_the_ancestors_when_the_generator_is_closed() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    path_registry = PathRegistry(registry)
    iterator = path_registry.iterate([[1, 2], [3, 4]])
    next(iterator)
    iterator.close()
    assert path_registry._ancestors == set()


def test_path_registry_initial_ancestors_are_copied_initial_ancestors() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    ancestors = {1}
    path_registry = PathRegistry(registry, ancestors)
    list(path_registry.iterate([1]))
    assert ancestors == {1}


def test_path_registry_initial_ancestors_are_copied_path_ancestors() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    ancestors = {1}
    path_registry = PathRegistry(registry, ancestors)
    list(path_registry.iterate([1]))
    assert path_registry._ancestors == {1}


def test_registry_iterate_wide_structure() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    data = [[i] for i in range(1_000)]
    assert sum(registry.iterate(data)) == 499500


def test_registry_iterate_first_traversal_yields_the_first_item() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    assert next(registry.iterate([[1, 2], [3, 4]])) == 1


def test_registry_iterate_second_traversal_is_independent() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    data = [[1, 2], [3, 4]]
    first = registry.iterate(data)
    second = registry.iterate(data)
    next(first)
    assert list(second) == [1, 2, 3, 4]


def test_registry_iterate_first_traversal_is_not_affected_by_the_second() -> None:
    registry = IteratorRegistry({object: DefaultIterator(), list: IterableIterator()})
    data = [[1, 2], [3, 4]]
    first = registry.iterate(data)
    second = registry.iterate(data)
    next(first)
    list(second)
    assert list(first) == [2, 3, 4]
