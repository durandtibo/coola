from __future__ import annotations

from collections import UserString, deque
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pytest

from coola.iterator import (
    BaseChildFinder,
    ChildFinderRegistry,
    DataclassChildFinder,
    DefaultChildFinder,
    IterableChildFinder,
    MappingChildFinder,
    bfs_iterate,
    dfs_iterate,
    get_default_child_finder_registry,
    get_default_registry,
    interface,
    register_child_finders,
)
from coola.registry import AlreadyRegisteredError
from tests.helpers import CONTAINER_SAMPLES, LEAF_SAMPLES


@dataclass
class Config:
    lr: float
    layers: list


class Pair:
    def __init__(self, left: Any, right: Any) -> None:
        self.left = left
        self.right = right


class PairChildFinder(BaseChildFinder[Pair]):
    def find_children(self, data: Pair) -> list:
        return [data.left, data.right]


@pytest.fixture(autouse=True)
def _reset_default_registry() -> Generator[None, None, None]:
    """Reset the registry before and after each test."""
    interface._default_registry.reset()
    yield
    interface._default_registry.reset()


@pytest.fixture
def registry() -> ChildFinderRegistry:
    return ChildFinderRegistry({object: DefaultChildFinder(), Pair: PairChildFinder()})


#################################
#     Tests for dfs_iterate     #
#################################


def test_dfs_iterate_default_registry() -> None:
    assert list(dfs_iterate({"a": 1, "b": [2, {"c": 3}], "d": 4})) == [1, 2, 3, 4]


def test_dfs_iterate_custom_registry(registry: ChildFinderRegistry) -> None:
    assert list(dfs_iterate(Pair(1, Pair(2, 3)), registry=registry)) == [1, 2, 3]


def test_dfs_iterate_registry_is_keyword_only() -> None:
    iterate: Any = dfs_iterate
    with pytest.raises(TypeError, match="positional"):
        iterate([1], ChildFinderRegistry())


def test_dfs_iterate_returns_a_generator() -> None:
    assert isinstance(dfs_iterate([1]), Iterator)


#################################
#     Tests for bfs_iterate     #
#################################


def test_bfs_iterate_default_registry() -> None:
    assert list(bfs_iterate({"a": 1, "b": [2, {"c": 3}], "d": 4})) == [1, 4, 2, 3]


def test_bfs_iterate_custom_registry(registry: ChildFinderRegistry) -> None:
    assert list(bfs_iterate(Pair(1, Pair(2, 3)), registry=registry)) == [1, 2, 3]


def test_bfs_iterate_registry_is_keyword_only() -> None:
    iterate: Any = bfs_iterate
    with pytest.raises(TypeError, match="positional"):
        iterate([1], ChildFinderRegistry())


def test_bfs_iterate_returns_a_generator() -> None:
    assert isinstance(bfs_iterate([1]), Iterator)


############################################
#     Tests for register_child_finders     #
############################################


def test_register_child_finders_registers_in_the_default_registry() -> None:
    register_child_finders({Pair: PairChildFinder()})
    assert get_default_registry().has_child_finder(Pair)


def test_register_child_finders_serves_both_orders() -> None:
    register_child_finders({Pair: PairChildFinder()})
    data = Pair(1, Pair(2, 3))
    assert (list(dfs_iterate(data)), list(bfs_iterate(data))) == ([1, 2, 3], [1, 2, 3])


def test_register_child_finders_already_registered_raises() -> None:
    with pytest.raises(AlreadyRegisteredError, match="already registered"):
        register_child_finders({list: IterableChildFinder()})


def test_register_child_finders_exist_ok_true() -> None:
    register_child_finders({list: DefaultChildFinder()}, exist_ok=True)
    assert list(dfs_iterate([1, 2])) == [[1, 2]]


def test_register_child_finders_exist_ok_is_keyword_only() -> None:
    register: Any = register_child_finders
    with pytest.raises(TypeError, match="positional"):
        register({list: IterableChildFinder()}, True)


##########################################
#     Tests for get_default_registry     #
##########################################


def test_get_default_registry_returns_registry() -> None:
    assert isinstance(get_default_registry(), ChildFinderRegistry)


def test_get_default_registry_returns_singleton() -> None:
    assert get_default_registry() is get_default_registry()


def test_get_default_child_finder_registry_is_an_alias() -> None:
    assert get_default_child_finder_registry is get_default_registry


@pytest.mark.parametrize("data_type", [object, str, bytes, int, float, complex, bool, UserString])
def test_default_registry_leaf_types_have_a_default_child_finder(data_type: type) -> None:
    assert type(get_default_registry().find_child_finder(data_type)) is DefaultChildFinder


@pytest.mark.parametrize("data_type", [list, tuple, deque, set, frozenset])
def test_default_registry_container_types_have_an_iterable_child_finder(data_type: type) -> None:
    assert type(get_default_registry().find_child_finder(data_type)) is IterableChildFinder


@pytest.mark.parametrize("data_type", [dict, MappingProxyType])
def test_default_registry_mapping_types_have_a_mapping_child_finder(data_type: type) -> None:
    assert type(get_default_registry().find_child_finder(data_type)) is MappingChildFinder


def test_default_registry_dataclass_has_a_dataclass_child_finder() -> None:
    assert type(get_default_registry().find_child_finder(Config)) is DataclassChildFinder


@pytest.mark.parametrize("iterate", [dfs_iterate, bfs_iterate])
@pytest.mark.parametrize("data", [UserString("abc"), "abc", b"abc", range(3)])
def test_default_registry_scalar_like_values_are_leaves(iterate: Any, data: Any) -> None:
    assert list(iterate([data])) == [data]


@pytest.mark.parametrize("iterate", [dfs_iterate, bfs_iterate])
def test_default_registry_iterators_are_leaves(iterate: Any) -> None:
    iterator = iter([1, 2])
    assert list(iterate([iterator, 3])) == [iterator, 3]


@pytest.mark.parametrize("iterate", [dfs_iterate, bfs_iterate])
def test_default_registry_enters_a_dataclass(iterate: Any) -> None:
    assert list(iterate(Config(0.1, [1, 2]))) == [0.1, 1, 2]


@pytest.mark.parametrize("iterate", [dfs_iterate, bfs_iterate])
def test_default_registry_dataclass_with_a_default_child_finder_is_a_leaf(iterate: Any) -> None:
    registry = get_default_registry().copy()
    registry.register(Config, DefaultChildFinder())
    assert list(iterate(Config(0.1, [1]), registry=registry)) == [Config(0.1, [1])]


################################################
#     Contract of the default child finders     #
################################################

SAMPLES = [
    *LEAF_SAMPLES,
    pytest.param(None, id="none"),
    *CONTAINER_SAMPLES,
    pytest.param(Config(0.1, [1]), id="dataclass"),
]


@pytest.mark.parametrize("data", SAMPLES)
def test_default_child_finder_is_pure(data: Any) -> None:
    finder = get_default_registry().find_child_finder(type(data))
    first = finder.find_children(data)
    second = finder.find_children(data)
    assert list(first or []) == list(second or [])


@pytest.mark.parametrize("data", SAMPLES)
def test_default_child_finder_result_is_none_or_iterable(data: Any) -> None:
    children = get_default_registry().find_child_finder(type(data)).find_children(data)
    assert children is None or isinstance(children, Iterable)
