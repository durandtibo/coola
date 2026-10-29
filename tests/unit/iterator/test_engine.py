from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

import pytest

from coola.iterator import (
    BaseChildFinder,
    ChildFinderRegistry,
    DefaultChildFinder,
    IterableChildFinder,
    MappingChildFinder,
    get_default_registry,
)
from coola.iterator.engine import breadth_first, depth_first, depth_first_items
from coola.registry import TypeNotRegisteredError
from tests.helpers import DEEP_NESTING_DEPTH, nest

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
    assert list(depth_first(nest(1, DEEP_NESTING_DEPTH), registry)) == [1]


###################################
#     Tests for breadth_first     #
###################################


def test_breadth_first_order(registry: ChildFinderRegistry) -> None:
    assert list(breadth_first([1, [2, [3]], 4, [5]], registry)) == [1, 4, 2, 5, 3]


def test_breadth_first_order_dict(registry: ChildFinderRegistry) -> None:
    assert list(breadth_first({"a": [1, 2], "b": 3, "c": {"d": 4}}, registry)) == [3, 1, 2, 4]


def test_breadth_first_deep_structure_does_not_raise(registry: ChildFinderRegistry) -> None:
    assert list(breadth_first(nest(1, DEEP_NESTING_DEPTH), registry)) == [1]


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


#######################################
#     Tests for depth_first_items     #
#######################################


@dataclass
class Item:
    a: int
    b: list


class Pt(NamedTuple):
    x: int
    y: int


class ProtocolChildFinder:
    r"""A child finder that follows the protocol: no ``find_items``."""

    def find_children(self, data: Any) -> Iterable[Any]:
        return [data.depth, data.depth + 1]


def test_depth_first_items_whole_output(registry: ChildFinderRegistry) -> None:
    data = {"a": 1, "b": [2, (3, {"c": 4})], "d": Item(5, [6])}
    assert list(depth_first_items(data, registry)) == [
        (("a",), 1),
        (("b", 0), 2),
        (("b", 1, 0), 3),
        (("b", 1, 1, "c"), 4),
        (("d", "a"), 5),
        (("d", "b", 0), 6),
    ]


def test_depth_first_items_root_leaf_has_an_empty_path(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items(42, registry)) == [((), 42)]


def test_depth_first_items_keeps_the_raw_keys(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items({1: {(2, 3): "x"}}, registry)) == [((1, (2, 3)), "x")]


def test_depth_first_items_set_label_is_the_element(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items({"a": {7}}, registry)) == [(("a", 7), 7)]


@pytest.mark.parametrize("data", [[], {}, [[], {}], {"a": {"b": []}}])
def test_depth_first_items_empty_container_yields_nothing(
    data: Any, registry: ChildFinderRegistry
) -> None:
    assert list(depth_first_items(data, registry)) == []


def test_depth_first_items_empty_root_with_empty(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items({}, registry, empty=True)) == [((), {})]


def test_depth_first_items_empty_containers_with_empty(registry: ChildFinderRegistry) -> None:
    data = {"a": {}, "b": [1, []], "c": ()}
    assert list(depth_first_items(data, registry, empty=True)) == [
        (("a",), {}),
        (("b", 0), 1),
        (("b", 1), []),
        (("c",), ()),
    ]


def test_depth_first_items_empty_dataclass_with_empty(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items({"a": Item(1, [])}, registry, empty=True)) == [
        (("a", "a"), 1),
        (("a", "b"), []),
    ]


def test_depth_first_items_same_leaves_as_depth_first(registry: ChildFinderRegistry) -> None:
    data = {"a": [1, {"b": 2}], "c": (3, [4, [5]]), "d": Item(6, [7])}
    assert [leaf for _, leaf in depth_first_items(data, registry)] == list(
        depth_first(data, registry)
    )


def test_depth_first_items_deep_structure_does_not_raise(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items(nest(1, DEEP_NESTING_DEPTH), registry)) == [
        ((0,) * DEEP_NESTING_DEPTH, 1)
    ]


def test_depth_first_items_cycle_raises(registry: ChildFinderRegistry) -> None:
    with pytest.raises(ValueError, match="Cycle detected"):
        list(depth_first_items(make_cycle(), registry))


def test_depth_first_items_cycle_yields_the_leaves_before_it(
    registry: ChildFinderRegistry,
) -> None:
    items = depth_first_items(make_cycle(), registry)
    assert next(items) == ((0,), 1)


def test_depth_first_items_shared_container_is_not_a_cycle(registry: ChildFinderRegistry) -> None:
    shared = [1]
    assert list(depth_first_items([shared, shared], registry)) == [((0, 0), 1), ((1, 0), 1)]


def test_depth_first_items_is_lazy(registry: ChildFinderRegistry) -> None:
    items = depth_first_items([1, make_cycle()], registry)
    assert next(items) == ((0,), 1)


def test_depth_first_items_does_not_modify_the_data(registry: ChildFinderRegistry) -> None:
    data = {"a": [1, {"b": 2}]}
    list(depth_first_items(data, registry))
    assert data == {"a": [1, {"b": 2}]}


def test_depth_first_items_protocol_child_finder_is_numbered(
    registry: ChildFinderRegistry,
) -> None:
    registry.register(Node, ProtocolChildFinder())
    assert list(depth_first_items({"n": Node(3)}, registry)) == [(("n", 0), 3), (("n", 1), 4)]


def test_depth_first_items_protocol_child_finder_leaf(registry: ChildFinderRegistry) -> None:
    registry.register(Node, NodeChildFinder())
    assert [path for path, _ in depth_first_items([Node(0)], registry)] == [(0,)]


def test_depth_first_items_base_child_finder_is_numbered(registry: ChildFinderRegistry) -> None:
    registry.register(Node, NodeChildFinder())
    assert [path for path, _ in depth_first_items(Node(2), registry)] == [(0, 0)]


def test_depth_first_items_failing_child_finder_raises(registry: ChildFinderRegistry) -> None:
    registry.register(Node, FailingChildFinder())
    with pytest.raises(RuntimeError, match="failing finder"):
        list(depth_first_items(Node(1), registry))


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param("abc", [((), "abc")], id="str root"),
        pytest.param(b"ab", [((), b"ab")], id="bytes root"),
        pytest.param(None, [((), None)], id="None root"),
        pytest.param(["abc"], [((0,), "abc")], id="str in a list"),
    ],
)
def test_depth_first_items_leaves(data: Any, expected: list, registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items(data, registry)) == expected


def test_depth_first_items_iterator_is_not_consumed(registry: ChildFinderRegistry) -> None:
    iterator = iter([1, 2])
    list(depth_first_items([iterator], registry))
    assert list(iterator) == [1, 2]


def test_depth_first_items_single_element(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items([1], registry)) == [((0,), 1)]


def test_depth_first_items_duplicate_values(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items({"a": 1, "b": 1}, registry)) == [(("a",), 1), (("b",), 1)]


def test_depth_first_items_tuple(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items((1, (2,)), registry)) == [((0,), 1), ((1, 0), 2)]


def test_depth_first_items_deque(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items(deque([1, 2]), registry)) == [((0,), 1), ((1,), 2)]


def test_depth_first_items_named_tuple(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items(Pt(1, 2), registry)) == [((0,), 1), ((1,), 2)]


def test_depth_first_items_range_is_a_leaf(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items(range(2), registry)) == [((), range(2))]


def test_depth_first_items_mapping_proxy(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items(MappingProxyType({"a": [1]}), registry)) == [(("a", 0), 1)]


def test_depth_first_items_ordered_dict_keeps_the_order(registry: ChildFinderRegistry) -> None:
    data = OrderedDict([("b", 1), ("a", 2)])
    assert list(depth_first_items(data, registry)) == [(("b",), 1), (("a",), 2)]


def test_depth_first_items_frozenset_label_is_the_element(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items(frozenset({"x"}), registry)) == [(("x",), "x")]


def test_depth_first_items_nested_dataclasses(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items(Item(Item(1, []), [2]), registry)) == [
        (("a", "a"), 1),
        (("b", 0), 2),
    ]


def test_depth_first_items_dataclass_class_is_a_leaf(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items([Item], registry)) == [((0,), Item)]


def test_depth_first_items_mixed_label_types(registry: ChildFinderRegistry) -> None:
    data = {1: [{"k": (2.5,)}], (1, 2): {True}}
    assert [path for path, _ in depth_first_items(data, registry)] == [
        (1, 0, "k", 0),
        ((1, 2), True),
    ]


def test_depth_first_items_path_is_a_tuple(registry: ChildFinderRegistry) -> None:
    assert isinstance(next(depth_first_items([[1]], registry))[0], tuple)


def test_depth_first_items_paths_are_independent(registry: ChildFinderRegistry) -> None:
    paths = [path for path, _ in depth_first_items([[1], [2]], registry)]
    assert paths == [(0, 0), (1, 0)]


def test_depth_first_items_is_deterministic(registry: ChildFinderRegistry) -> None:
    data = {"a": [1, {"b": 2}], "c": Item(3, [4])}
    assert list(depth_first_items(data, registry)) == list(depth_first_items(data, registry))


def test_depth_first_items_cycle_through_a_dict_raises(registry: ChildFinderRegistry) -> None:
    data: dict = {"a": 1}
    data["self"] = data
    with pytest.raises(ValueError, match="Cycle detected"):
        list(depth_first_items(data, registry))


def test_depth_first_items_cycle_through_a_dataclass_raises(
    registry: ChildFinderRegistry,
) -> None:
    item = Item(1, [])
    item.b.append(item)
    with pytest.raises(ValueError, match="Cycle detected"):
        list(depth_first_items(item, registry))


def test_depth_first_items_empty_with_empty_nested_levels(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items([[[]]], registry, empty=True)) == [((0, 0), [])]


def test_depth_first_items_empty_false_is_the_default(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items([[], 1], registry)) == list(
        depth_first_items([[], 1], registry, empty=False)
    )


def test_depth_first_items_empty_does_not_change_the_non_empty_leaves(
    registry: ChildFinderRegistry,
) -> None:
    data = {"a": [1, {}], "b": (2,)}
    leaves = [item for item in depth_first_items(data, registry, empty=True) if item[1] != {}]
    assert leaves == list(depth_first_items({"a": [1], "b": (2,)}, registry))


def test_depth_first_items_empty_string_is_a_leaf(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items([""], registry)) == [((0,), "")]


def test_depth_first_items_empty_root_leaf_with_empty(registry: ChildFinderRegistry) -> None:
    assert list(depth_first_items(1, registry, empty=True)) == [((), 1)]


def test_depth_first_items_empty_deep_structure_does_not_raise(
    registry: ChildFinderRegistry,
) -> None:
    assert list(depth_first_items(nest([], DEEP_NESTING_DEPTH), registry, empty=True)) == [
        ((0,) * DEEP_NESTING_DEPTH, [])
    ]


def test_depth_first_items_registered_leaf_type(registry: ChildFinderRegistry) -> None:
    registry.register(list, DefaultChildFinder(), exist_ok=True)
    assert list(depth_first_items({"a": [1]}, registry)) == [(("a",), [1])]
