from __future__ import annotations

from dataclasses import dataclass

import pytest

from coola.iterator.bfs import (
    ChildFinderRegistry,
    DataclassChildFinder,
    DefaultChildFinder,
    IterableChildFinder,
    MappingChildFinder,
)


@dataclass
class Point:
    x: float
    y: float = 0.0


@dataclass
class Container:
    items: list
    mapping: dict
    point: Point


@pytest.fixture
def registry() -> ChildFinderRegistry:
    return ChildFinderRegistry(
        {
            object: DefaultChildFinder(),
            list: IterableChildFinder(),
            dict: MappingChildFinder(),
            Point: DataclassChildFinder(),
            Container: DataclassChildFinder(),
        }
    )


#######################################
#     Tests for DataclassChildFinder  #
#######################################


def test_dataclass_child_finder_repr() -> None:
    assert repr(DataclassChildFinder()) == "DataclassChildFinder()"


def test_dataclass_child_finder_is_not_a_leaf() -> None:
    assert not DataclassChildFinder().is_leaf


def test_dataclass_child_finder_find_children() -> None:
    assert list(DataclassChildFinder().find_children(Point(1.0, 2.0))) == [1.0, 2.0]


def test_dataclass_child_finder_find_children_keeps_the_order_of_the_fields() -> None:
    assert list(DataclassChildFinder().find_children(Point(y=2.0, x=1.0))) == [1.0, 2.0]


def test_dataclass_child_finder_find_children_does_not_recurse() -> None:
    data = Container(items=[1], mapping={"a": 2}, point=Point(3))
    assert list(DataclassChildFinder().find_children(data)) == [[1], {"a": 2}, Point(3)]


def test_dataclass_child_finder_iterate_nested_structures(registry: ChildFinderRegistry) -> None:
    data = Container(items=[1, [2]], mapping={"a": 3}, point=Point(4, 5))
    assert list(registry.iterate(data)) == [1, 3, 4, 5, 2]


def test_dataclass_child_finder_iterate_dataclass_in_a_list(registry: ChildFinderRegistry) -> None:
    assert list(registry.iterate([Point(1.0), 2])) == [2, 1.0, 0.0]


def test_dataclass_child_finder_iterate_cycle_raises(registry: ChildFinderRegistry) -> None:
    data = Container(items=[], mapping={}, point=Point(1))
    data.items.append(data)
    with pytest.raises(ValueError, match="Cycle detected"):
        list(registry.iterate(data))
