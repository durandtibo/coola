from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from coola.recursive import (
    DefaultTransformer,
    MappingTransformer,
    SequenceTransformer,
    TransformerRegistry,
    recursive_apply,
)
from coola.recursive.pydantic import PydanticModelTransformer, get_default_transformers
from coola.testing.marks import pydantic_available
from coola.utils.imports import is_pydantic_available
from tests.unit.recursive.helpers import expand_not_none

if is_pydantic_available():
    from pydantic import BaseModel, ConfigDict, computed_field, field_validator

    class Point(BaseModel):
        x: Any
        y: Any = 0.0

    class Frozen(BaseModel):
        model_config = ConfigDict(frozen=True)

        value: Any

    class Extra(BaseModel):
        model_config = ConfigDict(extra="allow")

        a: Any

    class Nested(BaseModel):
        items: list
        mapping: dict
        point: Point

    class Computed(BaseModel):
        x: float

        @computed_field  # type: ignore[prop-decorator]
        @property
        def double(self) -> float:
            return self.x * 2

    class Validated(BaseModel):
        x: int

        @field_validator("x")
        @classmethod
        def check(cls, value: int) -> int:
            if value < 0:
                msg = "x must be positive"
                raise ValueError(msg)
            return value

    class SubPoint(Point):
        pass


@pytest.fixture
def registry() -> TransformerRegistry:
    transformer = PydanticModelTransformer()
    return TransformerRegistry(
        {
            object: DefaultTransformer(),
            list: SequenceTransformer(),
            dict: MappingTransformer(),
            BaseModel: transformer,
        }
    )


###############################################
#     Tests for PydanticModelTransformer      #
###############################################


def test_pydantic_model_transformer_repr() -> None:
    assert repr(PydanticModelTransformer()) == "PydanticModelTransformer()"


@pydantic_available
def test_pydantic_model_transformer_transform(registry: TransformerRegistry) -> None:
    assert registry.transform(Point(x=1.0, y=2.0), str) == Point(x="1.0", y="2.0")


@pydantic_available
def test_pydantic_model_transformer_transform_returns_the_same_class(
    registry: TransformerRegistry,
) -> None:
    assert type(registry.transform(SubPoint(x=1), str)) is SubPoint


@pydantic_available
def test_pydantic_model_transformer_transform_does_not_modify_the_input(
    registry: TransformerRegistry,
) -> None:
    point = Point(x=1, y=2)
    registry.transform(point, str)
    assert point == Point(x=1, y=2)


@pydantic_available
def test_pydantic_model_transformer_transform_returns_a_new_object(
    registry: TransformerRegistry,
) -> None:
    point = Point(x=1)
    assert registry.transform(point, lambda x: x) is not point


@pydantic_available
def test_pydantic_model_transformer_transform_nested(registry: TransformerRegistry) -> None:
    model = Nested(items=[1, 2], mapping={"a": 3}, point=Point(x=4))
    assert registry.transform(model, str).model_dump() == {
        "items": ["1", "2"],
        "mapping": {"a": "3"},
        "point": {"x": "4", "y": "0.0"},
    }


@pydantic_available
def test_pydantic_model_transformer_transform_frozen(registry: TransformerRegistry) -> None:
    assert registry.transform(Frozen(value=1), str) == Frozen(value="1")


@pydantic_available
def test_pydantic_model_transformer_transform_extra_fields(registry: TransformerRegistry) -> None:
    model = Extra(a=1, b=2)
    assert registry.transform(model, str).model_dump() == {"a": "1", "b": "2"}


@pydantic_available
def test_pydantic_model_transformer_transform_does_not_validate(
    registry: TransformerRegistry,
) -> None:
    """Test that the validators do not run again on the new values."""
    assert registry.transform(Validated(x=1), lambda _: -1).x == -1


@pydantic_available
def test_pydantic_model_transformer_computed_field_is_not_a_child() -> None:
    children, _ = expand_not_none(PydanticModelTransformer(), Computed(x=1.0))
    assert list(children) == [1.0]


@pydantic_available
def test_pydantic_model_transformer_computed_field_follows_the_new_value(
    registry: TransformerRegistry,
) -> None:
    assert registry.transform(Computed(x=1.0), lambda v: v + 1).double == 4.0


@pydantic_available
def test_pydantic_model_transformer_children_in_the_order_of_the_fields() -> None:
    children, _ = expand_not_none(PydanticModelTransformer(), Point(x=1, y=2))
    assert list(children) == [1, 2]


@pydantic_available
def test_pydantic_model_transformer_children_extra_fields_come_last() -> None:
    children, _ = expand_not_none(PydanticModelTransformer(), Extra(z=3, a=1))
    assert list(children) == [1, 3]


@pydantic_available
def test_pydantic_model_transformer_rebuild() -> None:
    _, rebuild = expand_not_none(PydanticModelTransformer(), Point(x=1, y=2))
    assert rebuild([3, 4]) == Point(x=3, y=4)


###############################################
#     Tests for get_default_transformers      #
###############################################


@pydantic_available
def test_get_default_transformers_keys() -> None:
    assert list(get_default_transformers()) == [BaseModel]


def test_get_default_transformers_without_pydantic() -> None:
    with patch("coola.recursive.pydantic.is_pydantic_available", lambda: False):
        assert get_default_transformers() == {}


@pydantic_available
def test_recursive_apply_enters_a_model() -> None:
    assert recursive_apply(Point(x=1, y=[2]), lambda v: v + 1 if isinstance(v, int) else v) == (
        Point(x=2, y=[3])
    )
