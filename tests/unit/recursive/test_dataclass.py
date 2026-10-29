from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any

import pytest

from coola.recursive import (
    DataclassTransformer,
    DefaultTransformer,
    MappingTransformer,
    SequenceTransformer,
    TransformerRegistry,
)
from tests.unit.recursive.helpers import expand_not_none


@dataclass
class Point:
    x: Any
    y: Any = 0.0


@dataclass(frozen=True)
class Frozen:
    value: Any


@dataclass
class Container:
    items: list
    mapping: dict
    point: Point


@dataclass
class WithDerived:
    x: float
    double: float = field(init=False)

    def __post_init__(self) -> None:
        self.double = self.x * 2


@dataclass(slots=True)
class Slotted:
    x: Any


@dataclass(frozen=True, slots=True)
class FrozenSlotted:
    a: Any
    b: Any


@dataclass
class WithInitVar:
    x: Any
    scale: InitVar[int]

    def __post_init__(self, scale: int) -> None:
        self.x = self.x * scale


@dataclass
class Empty:
    pass


@dataclass
class SubPoint(Point):
    pass


@dataclass
class CountInit:
    x: Any
    calls: list

    def __post_init__(self) -> None:
        self.calls.append(self.x)


@pytest.fixture
def registry() -> TransformerRegistry:
    return TransformerRegistry(
        {
            object: DefaultTransformer(),
            list: SequenceTransformer(),
            dict: MappingTransformer(),
            Point: DataclassTransformer(),
            Frozen: DataclassTransformer(),
            Container: DataclassTransformer(),
            WithDerived: DataclassTransformer(),
            Slotted: DataclassTransformer(),
            FrozenSlotted: DataclassTransformer(),
            WithInitVar: DataclassTransformer(),
            Empty: DataclassTransformer(),
            SubPoint: DataclassTransformer(),
            CountInit: DataclassTransformer(),
        }
    )


########################################
#     Tests for DataclassTransformer   #
########################################


def test_dataclass_transformer_repr() -> None:
    assert repr(DataclassTransformer()) == "DataclassTransformer()"


def test_dataclass_transformer_transform(registry: TransformerRegistry) -> None:
    assert registry.transform(Point(1.0, 2.0), str) == Point("1.0", "2.0")


def test_dataclass_transformer_transform_returns_the_same_class(
    registry: TransformerRegistry,
) -> None:
    assert type(registry.transform(Point(1.0), str)) is Point


def test_dataclass_transformer_transform_returns_a_new_object(
    registry: TransformerRegistry,
) -> None:
    point = Point(1.0)
    assert registry.transform(point, float) is not point


def test_dataclass_transformer_transform_does_not_change_the_input(
    registry: TransformerRegistry,
) -> None:
    point = Point(1.0, 2.0)
    registry.transform(point, str)
    assert point == Point(1.0, 2.0)


def test_dataclass_transformer_transform_frozen_dataclass(registry: TransformerRegistry) -> None:
    assert registry.transform(Frozen(1), str) == Frozen("1")


def test_dataclass_transformer_transform_slots_dataclass(registry: TransformerRegistry) -> None:
    assert registry.transform(Slotted(1), str) == Slotted("1")


def test_dataclass_transformer_transform_nested_structures(registry: TransformerRegistry) -> None:
    data = Container(items=[1, [2]], mapping={"a": 3}, point=Point(4, 5))
    assert registry.transform(data, str) == Container(
        items=["1", ["2"]], mapping={"a": "3"}, point=Point("4", "5")
    )


def test_dataclass_transformer_transform_field_that_is_a_dataclass_uses_the_registry(
    registry: TransformerRegistry,
) -> None:
    data = Frozen(Point(1, 2))
    assert registry.transform(data, str) == Frozen(Point("1", "2"))


def test_dataclass_transformer_transform_field_without_init_is_transformed(
    registry: TransformerRegistry,
) -> None:
    result = registry.transform(WithDerived(2.0), lambda x: x + 1)
    assert result.double == 5.0


def test_dataclass_transformer_transform_post_init_is_not_run(
    registry: TransformerRegistry,
) -> None:
    result = registry.transform(WithDerived(2.0), lambda x: x + 1)
    assert result.x == 3.0


def test_dataclass_transformer_transform_frozen_slotted_dataclass(
    registry: TransformerRegistry,
) -> None:
    assert registry.transform(FrozenSlotted(1, 2), str) == FrozenSlotted("1", "2")


def test_dataclass_transformer_transform_init_var_without_default(
    registry: TransformerRegistry,
) -> None:
    # ``__post_init__`` of the input set x to 2, and the copy gets the transformed field.
    assert registry.transform(WithInitVar(1, 2), str).x == "2"


def test_dataclass_transformer_transform_dataclass_without_field(
    registry: TransformerRegistry,
) -> None:
    assert registry.transform(Empty(), str) == Empty()


def test_dataclass_transformer_transform_subclass_keeps_the_type(
    registry: TransformerRegistry,
) -> None:
    assert type(registry.transform(SubPoint(1.0), str)) is SubPoint


def test_dataclass_transformer_transform_does_not_run_post_init(
    registry: TransformerRegistry,
) -> None:
    calls: list[Any] = []
    registry.transform(CountInit(1, calls), str)
    assert calls == [1]


def test_dataclass_transformer_transform_applies_func_to_the_fields_not_to_the_object(
    registry: TransformerRegistry,
) -> None:
    calls: list[Any] = []

    def record(value: Any) -> Any:
        calls.append(value)
        return value

    registry.transform(Point(1.0, 2.0), record)
    assert calls == [1.0, 2.0]


def test_dataclass_transformer_transform_dataclass_in_a_list(registry: TransformerRegistry) -> None:
    assert registry.transform([Point(1.0)], str) == [Point("1.0", "0.0")]


def test_dataclass_transformer_expand_children() -> None:
    children, _ = expand_not_none(DataclassTransformer(), Point(1.0, 2.0))
    assert children == [1.0, 2.0]


def test_dataclass_transformer_expand_rebuild() -> None:
    _, rebuild = expand_not_none(DataclassTransformer(), Point(1.0, 2.0))
    assert rebuild(["a", "b"]) == Point("a", "b")
