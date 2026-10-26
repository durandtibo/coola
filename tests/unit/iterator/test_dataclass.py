from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from coola.iterator import DataclassChildFinder


@dataclass
class Point:
    x: float
    y: float = 0.0


@dataclass(frozen=True)
class Frozen:
    a: int
    b: list


@dataclass(slots=True)
class Slotted:
    a: int
    b: int


@dataclass
class NoInit:
    a: int = field(init=False, default=3)


@dataclass
class Container:
    items: list
    point: Point


@dataclass
class Empty:
    pass


##########################################
#     Tests for DataclassChildFinder     #
##########################################


def test_dataclass_child_finder_repr() -> None:
    assert repr(DataclassChildFinder()) == "DataclassChildFinder()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(Point(1.0, 2.0), [1.0, 2.0], id="typical"),
        pytest.param(Point(y=2.0, x=1.0), [1.0, 2.0], id="order of the fields"),
        pytest.param(Frozen(1, [2]), [1, [2]], id="frozen"),
        pytest.param(Slotted(1, 2), [1, 2], id="slots"),
        pytest.param(NoInit(), [3], id="init=False"),
        pytest.param(Empty(), [], id="no field"),
        pytest.param(Container([1], Point(3)), [[1], Point(3)], id="does not recurse"),
    ],
)
def test_dataclass_child_finder_find_children(data: object, expected: list) -> None:
    assert list(DataclassChildFinder().find_children(data)) == expected


def test_dataclass_child_finder_find_children_does_not_modify_the_data() -> None:
    data = Point(1.0, 2.0)
    list(DataclassChildFinder().find_children(data))
    assert data == Point(1.0, 2.0)
