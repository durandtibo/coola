from __future__ import annotations

from collections import deque
from typing import NamedTuple

import pytest

from coola.iterator import IterableChildFinder


class Point(NamedTuple):
    x: int
    y: int


class CustomList(list):
    r"""Create a custom class that inherits from list."""


#########################################
#     Tests for IterableChildFinder     #
#########################################


def test_iterable_child_finder_repr() -> None:
    assert repr(IterableChildFinder()) == "IterableChildFinder()"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param([1, 2, 3], [1, 2, 3], id="list"),
        pytest.param((4, 2, 1), [4, 2, 1], id="tuple"),
        pytest.param([], [], id="empty list"),
        pytest.param([7], [7], id="single element"),
        pytest.param([1, 1, 1], [1, 1, 1], id="duplicates"),
        pytest.param([[1], [2]], [[1], [2]], id="does not recurse"),
        pytest.param({1}, [1], id="set"),
        pytest.param(frozenset([1]), [1], id="frozenset"),
        pytest.param(deque([1, 2]), [1, 2], id="deque"),
        pytest.param(Point(1, 2), [1, 2], id="named tuple"),
        pytest.param(CustomList([1, 2]), [1, 2], id="list subclass"),
    ],
)
def test_iterable_child_finder_find_children(data: object, expected: list) -> None:
    assert list(IterableChildFinder().find_children(data)) == expected


def test_iterable_child_finder_find_children_returns_the_container() -> None:
    data = [1, 2]
    assert IterableChildFinder().find_children(data) is data


def test_iterable_child_finder_find_children_does_not_modify_the_data() -> None:
    data = [1, [2]]
    list(IterableChildFinder().find_children(data))
    assert data == [1, [2]]
