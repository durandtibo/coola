r"""Integration tests that run ``dfs_iterate`` and ``bfs_iterate`` on one
table of inputs, and that compare the containers entered by ``iterator``
with the ones of ``recursive``."""

from __future__ import annotations

from collections import ChainMap, OrderedDict, deque
from collections.abc import Collection, Iterator, Set as AbstractSet
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NamedTuple

import pytest

from coola.iterator import bfs_iterate, dfs_iterate
from coola.recursive import recursive_apply

ITERATE = [dfs_iterate, bfs_iterate]


class Point(NamedTuple):
    x: int
    y: int


@dataclass
class Config:
    lr: float
    layers: list


class MyCollection(Collection):
    def __init__(self, items: list) -> None:
        self._items = items

    def __contains__(self, item: object) -> bool:
        return item in self._items

    def __iter__(self) -> Iterator:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


class MySet(AbstractSet):
    def __init__(self, items: list) -> None:
        self._items = items

    def __contains__(self, item: object) -> bool:
        return item in self._items

    def __iter__(self) -> Iterator:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


def make_cycle() -> list:
    data = [1]
    data.append(data)
    return data


SHARED = [1, 2]

#####################################
#     Leaves, containers, order     #
#####################################


@pytest.mark.parametrize("iterate", ITERATE)
@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(1, [1], id="int"),
        pytest.param(None, [None], id="none"),
        pytest.param("abc", ["abc"], id="str"),
        pytest.param(b"abc", [b"abc"], id="bytes"),
        pytest.param([], [], id="empty list"),
        pytest.param({}, [], id="empty dict"),
        pytest.param([[], {}, ()], [], id="nested empty containers"),
        pytest.param([1, [2, 3], {"a": 4}], [1, 2, 3, 4], id="nested"),
        pytest.param({"a": 1, "b": [2, 3]}, [1, 2, 3], id="dict values, keys are not yielded"),
        pytest.param((1, (2, 3)), [1, 2, 3], id="tuple"),
        pytest.param(deque([1, 2]), [1, 2], id="deque"),
        pytest.param(OrderedDict([("b", 1), ("a", 2)]), [1, 2], id="ordered dict"),
        pytest.param(MappingProxyType({"a": 1}), [1], id="mapping proxy"),
        pytest.param(frozenset([1]), [1], id="frozenset"),
        pytest.param({1}, [1], id="set"),
        pytest.param(Point(1, 2), [1, 2], id="named tuple"),
        pytest.param(Config(0.1, [1, 2]), [0.1, 1, 2], id="dataclass"),
        pytest.param([SHARED, SHARED], [1, 2, 1, 2], id="shared container"),
    ],
)
def test_iterate_leaves(iterate: Any, data: Any, expected: list) -> None:
    assert list(iterate(data)) == expected


def test_dfs_iterate_order() -> None:
    assert list(dfs_iterate([1, [2, [3]], 4, [5]])) == [1, 2, 3, 4, 5]


def test_bfs_iterate_order() -> None:
    assert list(bfs_iterate([1, [2, [3]], 4, [5]])) == [1, 4, 2, 5, 3]


@pytest.mark.parametrize("iterate", ITERATE)
def test_iterate_generator_is_a_leaf(iterate: Any) -> None:
    generator = iter([1, 2])
    assert next(iterate([generator])) is generator


@pytest.mark.parametrize("iterate", ITERATE)
def test_iterate_generator_is_not_consumed(iterate: Any) -> None:
    generator = iter([1, 2])
    list(iterate([generator]))
    assert list(generator) == [1, 2]


@pytest.mark.parametrize("iterate", ITERATE)
def test_iterate_does_not_modify_the_data(iterate: Any) -> None:
    data = {"a": [1, 2], "b": (3, {4})}
    list(iterate(data))
    assert data == {"a": [1, 2], "b": (3, {4})}


@pytest.mark.parametrize("iterate", ITERATE)
def test_iterate_is_lazy(iterate: Any) -> None:
    assert next(iterate([1, make_cycle()])) == 1


@pytest.mark.parametrize("iterate", ITERATE)
def test_iterate_cycle_raises(iterate: Any) -> None:
    with pytest.raises(ValueError, match="Cycle detected"):
        list(iterate(make_cycle()))


@pytest.mark.parametrize("iterate", ITERATE)
def test_iterate_cycle_through_a_dict_raises(iterate: Any) -> None:
    data: dict = {"a": 1}
    data["b"] = [data]
    with pytest.raises(ValueError, match="Cycle detected"):
        list(iterate(data))


def test_dfs_and_bfs_iterate_yield_the_same_leaves() -> None:
    data = {"a": [1, (2, {3})], "b": Config(0.5, [4]), "c": [[5], 6]}
    assert sorted(dfs_iterate(data), key=repr) == sorted(bfs_iterate(data), key=repr)


##################################################
#     Same containers as recursive_apply     #
##################################################

F8 = pytest.mark.xfail(
    reason="iterator enters every Collection, recursive only the Sequence (review/iterator.md F8)",
    strict=True,
)


def called_on(data: Any) -> list:
    calls = []
    recursive_apply(data, lambda x: calls.append(x) or x)
    return calls


@pytest.mark.parametrize(
    "data",
    [
        pytest.param([1, [2, 3]], id="list"),
        pytest.param((1, (2, 3)), id="tuple"),
        pytest.param(Point(1, 2), id="named tuple"),
        pytest.param(deque([1, 2]), id="deque"),
        pytest.param({"a": 1, "b": [2]}, id="dict"),
        pytest.param(OrderedDict(a=1), id="ordered dict"),
        pytest.param(MappingProxyType({"a": 1}), id="mapping proxy"),
        pytest.param(frozenset([1]), id="frozenset"),
        pytest.param({1}, id="set"),
        pytest.param(Config(0.1, [1, 2]), id="dataclass"),
        pytest.param("abc", id="str"),
        pytest.param(b"abc", id="bytes"),
        pytest.param(ChainMap({"a": 1}), id="chain map"),
        pytest.param(MySet([1]), id="user set", marks=F8),
        pytest.param(MyCollection([1]), id="user collection", marks=F8),
        pytest.param(range(3), id="range", marks=F8),
    ],
)
def test_dfs_iterate_visits_the_leaves_recursive_apply_calls_func_on(data: Any) -> None:
    assert list(dfs_iterate(data)) == called_on(data)
