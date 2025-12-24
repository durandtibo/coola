from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, NoReturn

import pytest

from coola import objects_are_equal
from coola.recursive import TransformerRegistry
from coola.recursive.transformer import MappingTransformer

if TYPE_CHECKING:
    from collections.abc import Callable


def test_mapping_transformer_repr() -> None:
    assert repr(MappingTransformer()) == "MappingTransformer()"


def test_mapping_transformer_str() -> None:
    assert str(MappingTransformer()) == "MappingTransformer()"


@pytest.mark.parametrize(
    ("data", "func", "expected"),
    [
        pytest.param({"a": 0}, str, {"a": "0"}, id="dict"),
        pytest.param({"a": 1, "b": 2}, str, {"a": "1", "b": "2"}, id="dict_str"),
        pytest.param({"a": 1, "b": 2}, lambda x: 2 * x, {"a": 2, "b": 4}, id="dict_mul"),
        pytest.param(
            OrderedDict({"a": 1, "b": 2}),
            str,
            OrderedDict({"a": "1", "b": "2"}),
            id="ordered_dict_str",
        ),
        pytest.param({}, str, {}, id="empty_dict"),
        pytest.param(OrderedDict({}), str, OrderedDict({}), id="empty_ordered_dict"),
    ],
)
def test_mapping_transformer_transform_parametrized(
    data: Any, func: Callable, expected: Any
) -> None:
    assert objects_are_equal(
        MappingTransformer().transform(data, func=func, registry=TransformerRegistry()), expected
    )


def test_mapping_transformer_transform_nested_dict() -> None:
    assert objects_are_equal(
        MappingTransformer().transform(
            {"a": 1, "b": {"x": 3, "y": 4}}, func=str, registry=TransformerRegistry()
        ),
        {"a": "1", "b": "{'x': 3, 'y': 4}"},
    )


def test_mapping_transformer_transform_with_custom_function() -> None:
    def custom_func(x: Any) -> str:
        return f"processed: {x}"

    assert objects_are_equal(
        MappingTransformer().transform(
            {"a": 1, "b": 2}, func=custom_func, registry=TransformerRegistry()
        ),
        {"a": "processed: 1", "b": "processed: 2"},
    )


def test_mapping_transformer_transform_with_identity_function() -> None:
    data = {"a": 1, "b": 2}
    result = MappingTransformer().transform(data, func=lambda x: x, registry=TransformerRegistry())
    assert result is not data


def test_mapping_transformer_transform_with_exception_in_function() -> None:
    def failing_func(x: Any) -> NoReturn:
        msg = f"Test error {x}"
        raise ValueError(msg)

    transformer = MappingTransformer()
    registry = TransformerRegistry()
    with pytest.raises(ValueError, match="Test error"):
        transformer.transform({"a": 1, "b": 2}, func=failing_func, registry=registry)


def test_mapping_transformer_transform_preserves_function_behavior() -> None:
    def create_list(x: Any) -> list:
        return [x, x + 1, x + 2]

    assert objects_are_equal(
        MappingTransformer().transform(
            {"a": 1, "b": 2}, func=create_list, registry=TransformerRegistry()
        ),
        {"a": [1, 2, 3], "b": [2, 3, 4]},
    )


def test_mapping_transformer_transform_with_stateful_function() -> None:
    counter = {"count": 0}

    def stateful_func(x: Any) -> Any:
        counter["count"] += 1
        return x * counter["count"]

    transformer = MappingTransformer()
    registry = TransformerRegistry()
    assert transformer.transform({"a": 1, "b": 2}, func=stateful_func, registry=registry) == {
        "a": 1,
        "b": 4,
    }
    assert transformer.transform({"a": 1, "b": 2}, func=stateful_func, registry=registry) == {
        "a": 3,
        "b": 8,
    }


def test_mapping_transformer_transform_with_closure() -> None:
    multiplier = 3

    def multiply_by(x: Any) -> Any:
        return x * multiplier

    assert MappingTransformer().transform(
        {"a": 1, "b": 2}, func=multiply_by, registry=TransformerRegistry()
    ) == {"a": 3, "b": 6}
