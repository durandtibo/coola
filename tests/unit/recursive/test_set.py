from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never

import pytest

from coola.equality import objects_are_equal
from coola.recursive import DefaultTransformer, SetTransformer, TransformerRegistry
from tests.unit.recursive.helpers import expand_not_none

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def registry() -> TransformerRegistry:
    transformer = SetTransformer()
    return TransformerRegistry(
        {object: DefaultTransformer(), set: transformer, frozenset: transformer}
    )


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
    assert objects_are_equal(registry.transform(data, func), expected)


def test_set_transformer_transform_nested_dict(registry: TransformerRegistry) -> None:
    assert objects_are_equal(
        registry.transform({1, frozenset({2, 3})}, str),
        {"1", frozenset({"2", "3"})},
    )


def test_set_transformer_transform_with_custom_function(registry: TransformerRegistry) -> None:
    def custom_func(x: Any) -> str:
        return f"processed: {x}"

    assert objects_are_equal(
        registry.transform({1, 2}, custom_func),
        {"processed: 1", "processed: 2"},
    )


def test_set_transformer_transform_with_identity_function(registry: TransformerRegistry) -> None:
    data = {1, 2}
    result = registry.transform(data, lambda x: x)
    assert result is not data


def test_set_transformer_transform_with_exception_in_function(
    registry: TransformerRegistry,
) -> None:
    def failing_func(x: Any) -> Never:
        msg = f"Test error {x}"
        raise ValueError(msg)

    with pytest.raises(ValueError, match=r"Test error"):
        registry.transform({1, 2}, failing_func)


def test_set_transformer_transform_preserves_function_behavior(
    registry: TransformerRegistry,
) -> None:
    def create_tuple(x: Any) -> tuple:
        return (x, x + 1, x + 2)

    assert objects_are_equal(
        registry.transform({1, 2}, create_tuple),
        {(1, 2, 3), (2, 3, 4)},
    )


def _make_stateful_func() -> Callable[[Any], Any]:
    counter = {"count": 0}

    def stateful_func(x: Any) -> Any:
        counter["count"] += 1
        return x * counter["count"]

    return stateful_func


def test_set_transformer_transform_with_stateful_function(registry: TransformerRegistry) -> None:
    assert registry.transform({1, 2}, _make_stateful_func()) == {1, 4}


def test_set_transformer_transform_with_stateful_function_second_call(
    registry: TransformerRegistry,
) -> None:
    stateful_func = _make_stateful_func()
    registry.transform({1, 2}, stateful_func)
    assert registry.transform({1, 2}, stateful_func) == {3, 8}


def test_set_transformer_transform_with_closure(registry: TransformerRegistry) -> None:
    multiplier = 3

    def multiply_by(x: Any) -> Any:
        return x * multiplier

    assert registry.transform({1, 2}, multiply_by) == {
        3,
        6,
    }


def test_set_transformer_expand_children() -> None:
    children, _ = expand_not_none(SetTransformer(), {1})
    assert children == [1]


def test_set_transformer_expand_rebuild_set() -> None:
    _, rebuild = expand_not_none(SetTransformer(), {1, 2})
    assert rebuild(["a", "b"]) == {"a", "b"}


def test_set_transformer_expand_rebuild_frozenset() -> None:
    _, rebuild = expand_not_none(SetTransformer(), frozenset({1, 2}))
    assert rebuild(["a", "b"]) == frozenset({"a", "b"})


def test_set_transformer_expand_rebuild_deduplicates() -> None:
    _, rebuild = expand_not_none(SetTransformer(), {1, 2, 3})
    assert rebuild([0, 0, 0]) == {0}


def test_set_transformer_expand_empty() -> None:
    children, rebuild = expand_not_none(SetTransformer(), set())
    assert (children, rebuild([])) == ([], set())


def test_set_transformer_expand_rebuild_unhashable_raises() -> None:
    _, rebuild = expand_not_none(SetTransformer(), {1})
    with pytest.raises(TypeError, match="unhashable type"):
        rebuild([[1]])


def test_set_transformer_expand_does_not_modify_the_input() -> None:
    data = {1, 2, 3}
    SetTransformer().expand(data)
    assert data == {1, 2, 3}
