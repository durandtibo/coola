from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, NamedTuple, Never

import pytest

from coola.equality import objects_are_equal
from coola.recursive import DefaultTransformer, SequenceTransformer, TransformerRegistry
from tests.unit.recursive.helpers import expand_not_none

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def registry() -> TransformerRegistry:
    sequence = SequenceTransformer()
    return TransformerRegistry(
        {object: DefaultTransformer(), list: sequence, tuple: sequence, deque: sequence}
    )


def test_sequence_transformer_repr() -> None:
    assert repr(SequenceTransformer()) == "SequenceTransformer()"


def test_sequence_transformer_str() -> None:
    assert str(SequenceTransformer()) == "SequenceTransformer()"


@pytest.mark.parametrize(
    ("data", "func", "expected"),
    [
        pytest.param([1, 2, 3], str, ["1", "2", "3"], id="str_list"),
        pytest.param((1, 2, 3), str, ("1", "2", "3"), id="str_tuple"),
        pytest.param([1, 2, 3], lambda x: x * 2, [2, 4, 6], id="multiply_int"),
        pytest.param([1.0, 2.0, 3.0], lambda x: x * 2, [2.0, 4.0, 6.0], id="multiply_float"),
        pytest.param(
            ["h", "e", "l", "l", "o"], str.upper, ["H", "E", "L", "L", "O"], id="str_upper"
        ),
        pytest.param([], str, [], id="empty_list"),
        pytest.param((), str, (), id="empty_tuple"),
    ],
)
def test_sequence_transformer_transform_parametrized(
    data: Any, func: Callable, expected: Any, registry: TransformerRegistry
) -> None:
    assert objects_are_equal(registry.transform(data, func), expected)


def test_sequence_transformer_transform_namedtuple(registry: TransformerRegistry) -> None:
    class Point(NamedTuple):
        x: int
        y: int
        z: int

    assert objects_are_equal(
        registry.transform(Point(x=1, y=2, z=3), lambda x: x * 2),
        Point(x=2, y=4, z=6),
    )


def test_sequence_transformer_transform_with_custom_function(registry: TransformerRegistry) -> None:
    def custom_func(x: Any) -> str:
        return f"processed: {x}"

    assert objects_are_equal(
        registry.transform([1, 2, 3], custom_func),
        ["processed: 1", "processed: 2", "processed: 3"],
    )


def test_sequence_transformer_transform_with_identity_function(
    registry: TransformerRegistry,
) -> None:
    data = [1, 2, 3]
    result = registry.transform(data, lambda x: x)
    assert result is not data


def test_sequence_transformer_transform_with_exception_in_function(
    registry: TransformerRegistry,
) -> None:
    def failing_func(x: Any) -> Never:
        msg = f"Test error {x}"
        raise ValueError(msg)

    with pytest.raises(ValueError, match=r"Test error"):
        registry.transform([1, 2, 3], failing_func)


def test_sequence_transformer_transform_preserves_function_behavior(
    registry: TransformerRegistry,
) -> None:
    def create_list(x: Any) -> list:
        return [x, x + 1, x + 2]

    assert objects_are_equal(
        registry.transform([1, 2, 3], create_list),
        [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
    )


def _make_stateful_func() -> Callable[[Any], Any]:
    counter = {"count": 0}

    def stateful_func(x: Any) -> Any:
        counter["count"] += 1
        return x * counter["count"]

    return stateful_func


def test_sequence_transformer_transform_with_stateful_function(
    registry: TransformerRegistry,
) -> None:
    assert registry.transform([1, 2, 3], _make_stateful_func()) == [
        1,
        4,
        9,
    ]


def test_sequence_transformer_transform_with_stateful_function_second_call(
    registry: TransformerRegistry,
) -> None:
    stateful_func = _make_stateful_func()
    registry.transform([1, 2, 3], stateful_func)
    assert registry.transform([1, 2, 3], stateful_func) == [4, 10, 18]


def test_sequence_transformer_transform_with_closure(registry: TransformerRegistry) -> None:
    multiplier = 3

    def multiply_by(x: Any) -> Any:
        return x * multiplier

    assert registry.transform([1, 2, 3], multiply_by) == [
        3,
        6,
        9,
    ]


def test_sequence_transformer_transform_deque_type(registry: TransformerRegistry) -> None:
    result = registry.transform(deque([1, 2], maxlen=4), lambda x: x + 1)
    assert type(result) is deque


def test_sequence_transformer_transform_deque_values(registry: TransformerRegistry) -> None:
    result = registry.transform(deque([1, 2], maxlen=4), lambda x: x + 1)
    assert result == deque([2, 3])


def test_sequence_transformer_transform_deque_maxlen(registry: TransformerRegistry) -> None:
    result = registry.transform(deque([1, 2], maxlen=4), lambda x: x + 1)
    assert result.maxlen == 4


def test_sequence_transformer_expand_children() -> None:
    children, _ = expand_not_none(SequenceTransformer(), (1, 2, 3))
    assert children == [1, 2, 3]


def test_sequence_transformer_expand_rebuild_list() -> None:
    _, rebuild = expand_not_none(SequenceTransformer(), [1, 2, 3])
    assert rebuild(["a", "b", "c"]) == ["a", "b", "c"]


def test_sequence_transformer_expand_rebuild_tuple() -> None:
    _, rebuild = expand_not_none(SequenceTransformer(), (1, 2))
    assert rebuild(["a", "b"]) == ("a", "b")


def test_sequence_transformer_expand_rebuild_deque_keeps_maxlen() -> None:
    _, rebuild = expand_not_none(SequenceTransformer(), deque([1, 2], maxlen=3))
    assert rebuild([3, 4]) == deque([3, 4], maxlen=3)


def test_sequence_transformer_expand_empty() -> None:
    children, rebuild = expand_not_none(SequenceTransformer(), [])
    assert (children, rebuild([])) == ([], [])


def test_sequence_transformer_expand_single_element() -> None:
    children, rebuild = expand_not_none(SequenceTransformer(), [1])
    assert (children, rebuild(["a"])) == ([1], ["a"])


def test_sequence_transformer_expand_duplicates() -> None:
    children, _ = expand_not_none(SequenceTransformer(), [1, 1, 1])
    assert children == [1, 1, 1]


def test_sequence_transformer_expand_does_not_modify_the_input() -> None:
    data = [1, 2, 3]
    SequenceTransformer().expand(data)
    assert data == [1, 2, 3]


def test_sequence_transformer_expand_rebuild_returns_a_new_container() -> None:
    data = [1, 2, 3]
    _, rebuild = expand_not_none(SequenceTransformer(), data)
    assert rebuild([1, 2, 3]) is not data
