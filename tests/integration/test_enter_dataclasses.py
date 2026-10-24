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
from coola.recursive import (
    TransformerRegistry,
    enter_dataclasses as enter_recursive,
    get_default_registry as get_recursive_registry,
    recursive_apply,
)


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
def transformer_registry() -> TransformerRegistry:
    registry = TransformerRegistry(dict(get_recursive_registry().items()))
    enter_recursive(registry=registry)
    return registry


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


def test_recursive_apply_dataclass_is_a_leaf_with_the_default_registry(model: Model) -> None:
    assert recursive_apply(model, type) is Model


def test_recursive_apply_enters_the_dataclasses(
    model: Model, transformer_registry: TransformerRegistry
) -> None:
    result = recursive_apply(
        model, lambda x: x * 2 if isinstance(x, int) else x, registry=transformer_registry
    )
    assert result == Model("net", [Layer(16), Layer(8, "tanh")], {"lr": 0.1, "shape": (4, 6)})


def test_recursive_apply_dataclass_that_is_a_dict_value(
    transformer_registry: TransformerRegistry,
) -> None:
    result = recursive_apply({"layer": Layer(2)}, str, registry=transformer_registry)
    assert result == {"layer": Layer("2", "relu")}


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
