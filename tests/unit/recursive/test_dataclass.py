from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from coola.recursive import (
    DataclassTransformer,
    DefaultTransformer,
    MappingTransformer,
    SequenceTransformer,
    TransformerRegistry,
)


@dataclass
class Point:
    x: float
    y: float = 0.0


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
    x: float


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
        }
    )


########################################
#     Tests for DataclassTransformer   #
########################################


def test_dataclass_transformer_repr() -> None:
    assert repr(DataclassTransformer()) == "DataclassTransformer()"


def test_dataclass_transformer_transform(registry: TransformerRegistry) -> None:
    assert DataclassTransformer().transform(Point(1.0, 2.0), str, registry) == Point("1.0", "2.0")


def test_dataclass_transformer_transform_returns_the_same_class(
    registry: TransformerRegistry,
) -> None:
    assert type(DataclassTransformer().transform(Point(1.0), str, registry)) is Point


def test_dataclass_transformer_transform_returns_a_new_object(
    registry: TransformerRegistry,
) -> None:
    point = Point(1.0)
    assert DataclassTransformer().transform(point, float, registry) is not point


def test_dataclass_transformer_transform_does_not_change_the_input(
    registry: TransformerRegistry,
) -> None:
    point = Point(1.0, 2.0)
    DataclassTransformer().transform(point, str, registry)
    assert point == Point(1.0, 2.0)


def test_dataclass_transformer_transform_frozen_dataclass(registry: TransformerRegistry) -> None:
    assert DataclassTransformer().transform(Frozen(1), str, registry) == Frozen("1")


def test_dataclass_transformer_transform_slots_dataclass(registry: TransformerRegistry) -> None:
    assert DataclassTransformer().transform(Slotted(1), str, registry) == Slotted("1")


def test_dataclass_transformer_transform_nested_structures(registry: TransformerRegistry) -> None:
    data = Container(items=[1, [2]], mapping={"a": 3}, point=Point(4, 5))
    assert DataclassTransformer().transform(data, str, registry) == Container(
        items=["1", ["2"]], mapping={"a": "3"}, point=Point("4", "5")
    )


def test_dataclass_transformer_transform_field_that_is_a_dataclass_uses_the_registry(
    registry: TransformerRegistry,
) -> None:
    data = Frozen(Point(1, 2))
    assert DataclassTransformer().transform(data, str, registry) == Frozen(Point("1", "2"))


def test_dataclass_transformer_transform_field_without_init_is_recomputed(
    registry: TransformerRegistry,
) -> None:
    result = DataclassTransformer().transform(WithDerived(2.0), lambda x: x + 1, registry)
    assert result.double == 6.0


def test_dataclass_transformer_transform_applies_func_to_the_fields_not_to_the_object(
    registry: TransformerRegistry,
) -> None:
    calls: list[Any] = []

    def record(value: Any) -> Any:
        calls.append(value)
        return value

    DataclassTransformer().transform(Point(1.0, 2.0), record, registry)
    assert calls == [1.0, 2.0]


def test_dataclass_transformer_transform_dataclass_in_a_list(registry: TransformerRegistry) -> None:
    assert registry.transform([Point(1.0)], str) == [Point("1.0", "0.0")]
