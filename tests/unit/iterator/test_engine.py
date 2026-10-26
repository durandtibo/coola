from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from coola.iterator import (
    BaseChildFinder,
    ChildFinderRegistry,
    DefaultChildFinder,
    IterableChildFinder,
    MappingChildFinder,
    get_default_registry,
)
from coola.iterator.engine import breadth_first, depth_first
from coola.registry import TypeNotRegisteredError
from tests.helpers import nest

if TYPE_CHECKING:
    from collections.abc import Iterable

ENGINES = [depth_first, breadth_first]


class Node:
    r"""A node that builds a fresh child for each call of the child
    finder."""

    def __init__(self, depth: int) -> None:
        self.depth = depth


class NodeChildFinder(BaseChildFinder[Node]):
    def find_children(self, data: Node) -> Iterable[Any] | None:
        return [Node(data.depth - 1)] if data.depth > 0 else None


class SelfChildFinder(BaseChildFinder[Any]):
    r"""A child finder that returns the object itself as a child."""

    def find_children(self, data: Any) -> Iterable[Any]:
        return [data]


class FailingChildFinder(BaseChildFinder[Any]):
    def find_children(self, data: Any) -> Iterable[Any]:
        del data
        msg = "failing finder"
        raise RuntimeError(msg)


@pytest.fixture
def registry() -> ChildFinderRegistry:
    return get_default_registry().copy()


def make_cycle() -> list:
    data = [1]
    data.append(data)
    return data


#################################
#     Tests for depth_first     #
#################################


def test_depth_first_order(registry: ChildFinderRegistry) -> None:
    assert list(depth_first([1, [2, [3]], 4, [5]], registry)) == [1, 2, 3, 4, 5]


def test_depth_first_order_dict_tuple_set(registry: ChildFinderRegistry) -> None:
    assert list(depth_first({"a": (1, {2}), "b": [3]}, registry)) == [1, 2, 3]


def test_depth_first_deep_structure_does_not_raise(registry: ChildFinderRegistry) -> None:
    assert list(depth_first(nest(1, 100_000), registry)) == [1]


###################################
#     Tests for breadth_first     #
###################################


def test_breadth_first_order(registry: ChildFinderRegistry) -> None:
    assert list(breadth_first([1, [2, [3]], 4, [5]], registry)) == [1, 4, 2, 5, 3]


def test_breadth_first_order_dict(registry: ChildFinderRegistry) -> None:
    assert list(breadth_first({"a": [1, 2], "b": 3, "c": {"d": 4}}, registry)) == [3, 1, 2, 4]


def test_breadth_first_deep_structure_does_not_raise(registry: ChildFinderRegistry) -> None:
    assert list(breadth_first(nest(1, 100_000), registry)) == [1]


def test_breadth_first_chain_of_temporary_nodes_is_not_a_cycle(
    registry: ChildFinderRegistry,
) -> None:
    # Regression: the ancestors were kept as ids, and the id of a freed temporary node was reused.
    registry.register(Node, NodeChildFinder())
    assert len(list(breadth_first(Node(50), registry))) == 1


##################################
#     Tests for both engines     #
##################################


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_root_leaf_yields_itself(engine: Any, registry: ChildFinderRegistry) -> None:
    assert list(engine(42, registry)) == [42]


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_none_is_a_leaf(engine: Any, registry: ChildFinderRegistry) -> None:
    assert list(engine([None], registry)) == [None]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("data", [[], {}, [[], {}, ()]])
def test_engine_empty_container_yields_nothing(
    engine: Any, data: Any, registry: ChildFinderRegistry
) -> None:
    assert list(engine(data, registry)) == []


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_shared_container_gives_its_leaves_twice(
    engine: Any, registry: ChildFinderRegistry
) -> None:
    shared = [1, 2]
    assert list(engine([shared, shared], registry)) == [1, 2, 1, 2]


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_cycle_through_a_list_raises(engine: Any, registry: ChildFinderRegistry) -> None:
    with pytest.raises(ValueError, match="Cycle detected"):
        list(engine(make_cycle(), registry))


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_cycle_through_a_dict_raises(engine: Any, registry: ChildFinderRegistry) -> None:
    data: dict = {}
    data["a"] = data
    with pytest.raises(ValueError, match="Cycle detected"):
        list(engine(data, registry))


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_cycle_through_a_tuple_in_a_list_raises(
    engine: Any, registry: ChildFinderRegistry
) -> None:
    data: list = []
    data.append((data,))
    with pytest.raises(ValueError, match="Cycle detected"):
        list(engine(data, registry))


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_cycle_with_a_custom_finder_raises(
    engine: Any, registry: ChildFinderRegistry
) -> None:
    registry.register(Node, SelfChildFinder())
    with pytest.raises(ValueError, match="Cycle detected"):
        list(engine(Node(1), registry))


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_leaves_before_a_cycle_are_yielded(
    engine: Any, registry: ChildFinderRegistry
) -> None:
    assert next(engine([1, make_cycle()], registry)) == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_nothing_is_walked_before_the_first_next(
    engine: Any, registry: ChildFinderRegistry
) -> None:
    registry.register(Node, FailingChildFinder())
    iterator = engine([Node(1)], registry)  # the generator is not started: no error
    assert iterator is not None


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_exception_of_a_finder_propagates(
    engine: Any, registry: ChildFinderRegistry
) -> None:
    registry.register(Node, FailingChildFinder())
    with pytest.raises(RuntimeError, match="failing finder"):
        list(engine([Node(1)], registry))


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_works_after_an_exception(engine: Any, registry: ChildFinderRegistry) -> None:
    with pytest.raises(ValueError, match="Cycle detected"):
        list(engine(make_cycle(), registry))
    assert list(engine([1, [2]], registry)) == [1, 2]


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_works_after_an_early_break(engine: Any, registry: ChildFinderRegistry) -> None:
    data = [[1, 2], [3]]
    for _ in engine(data, registry):
        break
    assert list(engine(data, registry)) == [1, 2, 3]


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_interleaved_traversals_are_independent(
    engine: Any, registry: ChildFinderRegistry
) -> None:
    first = engine([1, [2, 3]], registry)
    second = engine([4, [5, 6]], registry)
    assert [next(first), next(second), next(first), next(second)] == [1, 4, 2, 5]


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_finder_registered_during_a_traversal_is_not_used(
    engine: Any, registry: ChildFinderRegistry
) -> None:
    data = [1, [2, 3]]
    iterator = engine(data, registry)
    next(iterator)
    registry.register(list, DefaultChildFinder(), exist_ok=True)
    assert list(iterator) == [2, 3]


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_unregistered_type_raises(engine: Any) -> None:
    registry = ChildFinderRegistry({list: IterableChildFinder(), dict: MappingChildFinder()})
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        list(engine([1], registry))


@pytest.mark.parametrize("engine", ENGINES)
def test_engine_finder_with_only_find_children_is_accepted(
    engine: Any, registry: ChildFinderRegistry
) -> None:
    registry.register(Node, NodeChildFinder())
    assert [type(leaf) for leaf in engine(Node(2), registry)] == [Node]
