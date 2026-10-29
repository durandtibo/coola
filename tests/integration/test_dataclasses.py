from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from coola.iterator import DefaultChildFinder, bfs_iterate, dfs_iterate, get_default_registry
from coola.recursive import recursive_apply


@dataclass
class Layer:
    size: Any
    activation: str = "relu"


@dataclass
class Model:
    name: str
    layers: list
    options: dict[str, Any]


@pytest.fixture
def model() -> Model:
    return Model("net", [Layer(8), Layer(4, "tanh")], {"lr": 0.1, "shape": (2, 3)})


def test_recursive_apply_enters_the_dataclasses(model: Model) -> None:
    result = recursive_apply(model, lambda x: x * 2 if isinstance(x, int) else x)
    assert result == Model("net", [Layer(16), Layer(8, "tanh")], {"lr": 0.1, "shape": (4, 6)})


def test_recursive_apply_dataclass_that_is_a_dict_value() -> None:
    assert recursive_apply({"layer": Layer(2)}, str) == {"layer": Layer("2", "relu")}


def test_dfs_iterate_enters_the_dataclasses(model: Model) -> None:
    assert list(dfs_iterate(model)) == ["net", 8, "relu", 4, "tanh", 0.1, 2, 3]


def test_dfs_iterate_dataclass_with_its_own_child_finder_is_a_leaf(model: Model) -> None:
    registry = get_default_registry().copy()
    registry.register(Layer, DefaultChildFinder())
    assert list(dfs_iterate(model, registry=registry)) == [
        "net",
        Layer(8),
        Layer(4, "tanh"),
        0.1,
        2,
        3,
    ]


def test_bfs_iterate_enters_the_dataclasses(model: Model) -> None:
    assert list(bfs_iterate(model)) == ["net", 0.1, 8, "relu", 4, "tanh", 2, 3]


def test_bfs_iterate_dataclass_with_its_own_child_finder_is_a_leaf(model: Model) -> None:
    registry = get_default_registry().copy()
    registry.register(Layer, DefaultChildFinder())
    assert list(bfs_iterate(model, registry=registry)) == [
        "net",
        Layer(8),
        Layer(4, "tanh"),
        0.1,
        2,
        3,
    ]
