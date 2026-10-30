from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any

from coola.jsonable import to_jsonable
from coola.nested import remove_keys_if
from coola.recursive import DefaultTransformer, get_default_registry, recursive_apply
from coola.rounding import round_values


@dataclass(frozen=True, slots=True)
class Frozen:
    a: Any
    b: Any


@dataclass
class Derived:
    x: float
    double: float = field(init=False)

    def __post_init__(self) -> None:
        self.double = self.x * 2


@dataclass
class WithInitVar:
    x: Any
    scale: InitVar[int]


@dataclass
class Node:
    name: str
    children: list[Any]


def test_recursive_apply_frozen_slotted_dataclass() -> None:
    assert recursive_apply(Frozen(1, [2]), str) == Frozen("1", ["2"])


def test_recursive_apply_transforms_the_field_without_init() -> None:
    assert recursive_apply(Derived(2.0), lambda x: x + 1).double == 5.0


def test_recursive_apply_init_var_without_default() -> None:
    assert recursive_apply(WithInitVar(1, 3), str).x == "1"


def test_recursive_apply_nested_dataclasses() -> None:
    data = Node("root", [Node("a", []), {"b": Node("b", [])}])
    assert recursive_apply(data, str.upper) == Node("ROOT", [Node("A", []), {"b": Node("B", [])}])


def test_recursive_apply_dataclass_registered_with_its_own_transformer_has_priority() -> None:
    registry = get_default_registry().copy()
    registry.register(Frozen, DefaultTransformer())
    assert recursive_apply(Frozen(1, 2), type, registry=registry) is Frozen


def test_recursive_apply_does_not_modify_the_dataclass() -> None:
    data = Node("root", [1])
    recursive_apply(data, str)
    assert data == Node("root", [1])


def test_round_values_recursive_dataclass() -> None:
    assert round_values(Node("a", [1.234]), decimals=1, recursive=True) == Node("a", [1.2])


def test_to_jsonable_dataclass_is_converted_whole() -> None:
    assert to_jsonable(Frozen(1, {2})) == {"a": 1, "b": [2]}


def test_remove_keys_if_enters_the_dataclass() -> None:
    node = Node("a", [{"key": 1, "other": 2}])
    assert remove_keys_if({"x": node}, lambda key: key == "key") == {"x": Node("a", [{"other": 2}])}
