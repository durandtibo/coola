from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

import pytest

from coola.equality import objects_are_equal
from coola.recursive import DefaultTransformer, SetTransformer, TransformerRegistry

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def registry() -> TransformerRegistry:
    return TransformerRegistry({object: DefaultTransformer()})


def test_set_transformer_repr() -> None:
    assert repr(SetTransformer()) == "SetTransformer()"


def test_set_transformer_str() -> None:
    assert str(SetTransformer()) == "SetTransformer()"


@pytest.mark.parametrize(
    ("data", "func", "expected"),
    [
        pytest.param({0}, str, {"0"}, id="set"),
        pytest.param({1, 2}, str, {"1", "2"}, id="set_str"),
        pytest.param({1, 2}, lambda x: 2 * x, {2, 4}, id="set_mul"),
        pytest.param(
            frozenset({1, 2}),
            str,
            frozenset({"1", "2"}),
            id="frozenset_str",
        ),
        pytest.param(set(), str, set(), id="empty_set"),
        pytest.param(frozenset(), str, frozenset(), id="empty_frozenset"),
    ],
)
def test_set_transformer_transform_parametrized(
    data: Any, func: Callable, expected: Any, registry: TransformerRegistry
) -> None:
    assert objects_are_equal(
        SetTransformer().transform(data, func=func, registry=registry), expected
    )


def test_set_transformer_transform_nested_dict(registry: TransformerRegistry) -> None:
    assert objects_are_equal(
        SetTransformer().transform({1, frozenset({2, 3})}, func=str, registry=registry),
        {"1", "frozenset({2, 3})"},
    )


def test_set_transformer_transform_with_custom_function(registry: TransformerRegistry) -> None:
    def custom_func(x: Any) -> str:
        return f"processed: {x}"

    assert objects_are_equal(
        SetTransformer().transform({1, 2}, func=custom_func, registry=registry),
        {"processed: 1", "processed: 2"},
    )


def test_set_transformer_transform_with_identity_function(registry: TransformerRegistry) -> None:
    data = {1, 2}
    result = SetTransformer().transform(data, func=lambda x: x, registry=registry)
    assert result is not data


def test_set_transformer_transform_with_exception_in_function(
    registry: TransformerRegistry,
) -> None:
    def failing_func(x: Any) -> NoReturn:
        msg = f"Test error {x}"
        raise ValueError(msg)

    transformer = SetTransformer()
    with pytest.raises(ValueError, match="Test error"):
        transformer.transform({1, 2}, func=failing_func, registry=registry)


def test_set_transformer_transform_preserves_function_behavior(
    registry: TransformerRegistry,
) -> None:
    def create_tuple(x: Any) -> tuple:
        return (x, x + 1, x + 2)

    assert objects_are_equal(
        SetTransformer().transform({1, 2}, func=create_tuple, registry=registry),
        {(1, 2, 3), (2, 3, 4)},
    )


def test_set_transformer_transform_with_stateful_function(registry: TransformerRegistry) -> None:
    counter = {"count": 0}

    def stateful_func(x: Any) -> Any:
        counter["count"] += 1
        return x * counter["count"]

    transformer = SetTransformer()
    assert transformer.transform({1, 2}, func=stateful_func, registry=registry) == {1, 4}
    assert transformer.transform({1, 2}, func=stateful_func, registry=registry) == {3, 8}


def test_set_transformer_transform_with_closure(registry: TransformerRegistry) -> None:
    multiplier = 3

    def multiply_by(x: Any) -> Any:
        return x * multiplier

    assert SetTransformer().transform({1, 2}, func=multiply_by, registry=registry) == {
        3,
        6,
    }
