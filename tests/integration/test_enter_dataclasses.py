from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from coola.iterator.bfs import (
    ChildFinderRegistry,
    bfs_iterate,
    enter_dataclasses as enter_bfs,
    get_default_registry as get_bfs_registry,
)
from coola.iterator.dfs import (
    IteratorRegistry,
    dfs_iterate,
    enter_dataclasses as enter_dfs,
    get_default_registry as get_dfs_registry,
)
from coola.recursive import recursive_apply


@dataclass
class Layer:
    size: int
    activation: str = "relu"


@dataclass
class Model:
    name: str
    layers: list
    options: dict[str, Any]


@pytest.fixture
def model() -> Model:
    return Model("net", [Layer(8), Layer(4, "tanh")], {"lr": 0.1, "shape": (2, 3)})


# The registries are copies of the default registries (their real entries), so the default
# registries of the process are not changed.


@pytest.fixture
def dfs_registry() -> IteratorRegistry:
    registry = IteratorRegistry(dict(get_dfs_registry().items()))
    enter_dfs(registry=registry)
    return registry


@pytest.fixture
def bfs_registry() -> ChildFinderRegistry:
    registry = ChildFinderRegistry(dict(get_bfs_registry().items()))
    enter_bfs(registry=registry)
    return registry


def test_recursive_apply_enters_the_dataclasses(model: Model) -> None:
    result = recursive_apply(model, lambda x: x * 2 if isinstance(x, int) else x)
    assert result == Model("net", [Layer(16), Layer(8, "tanh")], {"lr": 0.1, "shape": (4, 6)})


def test_recursive_apply_dataclass_that_is_a_dict_value() -> None:
    assert recursive_apply({"layer": Layer(2)}, str) == {"layer": Layer("2", "relu")}


def test_dfs_iterate_dataclass_is_a_leaf_with_the_default_registry(model: Model) -> None:
    assert list(dfs_iterate(model)) == [model]


def test_dfs_iterate_enters_the_dataclasses(model: Model, dfs_registry: IteratorRegistry) -> None:
    assert list(dfs_iterate(model, registry=dfs_registry)) == [
        "net",
        8,
        "relu",
        4,
        "tanh",
        0.1,
        2,
        3,
    ]


def test_bfs_iterate_dataclass_is_a_leaf_with_the_default_registry(model: Model) -> None:
    assert list(bfs_iterate(model)) == [model]


def test_bfs_iterate_enters_the_dataclasses(
    model: Model, bfs_registry: ChildFinderRegistry
) -> None:
    assert list(bfs_iterate(model, registry=bfs_registry)) == [
        "net",
        0.1,
        8,
        "relu",
        4,
        "tanh",
        2,
        3,
    ]
