from __future__ import annotations

from dataclasses import dataclass

import pytest

from coola.iterator.dfs import (
    DataclassIterator,
    DefaultIterator,
    IterableIterator,
    IteratorRegistry,
    MappingIterator,
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
def registry() -> IteratorRegistry:
    return IteratorRegistry(
        {
            object: DefaultIterator(),
            list: IterableIterator(),
            dict: MappingIterator(),
            Point: DataclassIterator(),
            Container: DataclassIterator(),
        }
    )


####################################
#     Tests for DataclassIterator  #
####################################


def test_dataclass_iterator_repr() -> None:
    assert repr(DataclassIterator()) == "DataclassIterator()"


def test_dataclass_iterator_iterate(registry: IteratorRegistry) -> None:
    assert list(DataclassIterator().iterate(Point(1.0, 2.0), registry)) == [1.0, 2.0]


def test_dataclass_iterator_iterate_keeps_the_order_of_the_fields(
    registry: IteratorRegistry,
) -> None:
    assert list(DataclassIterator().iterate(Point(y=2.0, x=1.0), registry)) == [1.0, 2.0]


def test_dataclass_iterator_iterate_nested_structures(registry: IteratorRegistry) -> None:
    data = Container(items=[1, [2]], mapping={"a": 3}, point=Point(4, 5))
    assert list(DataclassIterator().iterate(data, registry)) == [1, 2, 3, 4, 5]


def test_dataclass_iterator_iterate_dataclass_in_a_list(registry: IteratorRegistry) -> None:
    assert list(registry.iterate([Point(1.0), 2])) == [1.0, 0.0, 2]


def test_dataclass_iterator_iterate_cycle_raises(registry: IteratorRegistry) -> None:
    data = Container(items=[], mapping={}, point=Point(1))
    data.items.append(data)
    with pytest.raises(ValueError, match="Cycle detected"):
        list(registry.iterate(data))
