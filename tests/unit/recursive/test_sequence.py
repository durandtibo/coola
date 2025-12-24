from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn

import pytest

from coola import objects_are_equal
from coola.recursive import TransformerRegistry
from coola.recursive.transformer import SequenceTransformer

if TYPE_CHECKING:
    from collections.abc import Callable


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
    data: Any, func: Callable, expected: Any
) -> None:
    assert objects_are_equal(
        SequenceTransformer().transform(data, func=func, registry=TransformerRegistry()), expected
    )


def test_sequence_transformer_transform_namedtuple() -> None:
    class Point(NamedTuple):
        x: int
        y: int
        z: int

    assert objects_are_equal(
        SequenceTransformer().transform(
            Point(x=1, y=2, z=3), func=lambda x: x * 2, registry=TransformerRegistry()
        ),
        Point(x=2, y=4, z=6),
    )


def test_sequence_transformer_transform_with_custom_function() -> None:
    def custom_func(x: Any) -> str:
        return f"processed: {x}"

    assert objects_are_equal(
        SequenceTransformer().transform(
            [1, 2, 3], func=custom_func, registry=TransformerRegistry()
        ),
        ["processed: 1", "processed: 2", "processed: 3"],
    )


def test_sequence_transformer_transform_with_identity_function() -> None:
    data = [1, 2, 3]
    result = SequenceTransformer().transform(data, func=lambda x: x, registry=TransformerRegistry())
    assert result is not data


def test_sequence_transformer_transform_with_exception_in_function() -> None:
    def failing_func(x: Any) -> NoReturn:
        msg = f"Test error {x}"
        raise ValueError(msg)

    transformer = SequenceTransformer()
    registry = TransformerRegistry()
    with pytest.raises(ValueError, match="Test error"):
        transformer.transform([1, 2, 3], func=failing_func, registry=registry)


def test_sequence_transformer_transform_preserves_function_behavior() -> None:
    def create_list(x: Any) -> list:
        return [x, x + 1, x + 2]

    assert objects_are_equal(
        SequenceTransformer().transform(
            [1, 2, 3], func=create_list, registry=TransformerRegistry()
        ),
        [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
    )


def test_sequence_transformer_transform_with_stateful_function() -> None:
    counter = {"count": 0}

    def stateful_func(x: Any) -> Any:
        counter["count"] += 1
        return x * counter["count"]

    transformer = SequenceTransformer()
    registry = TransformerRegistry()
    assert transformer.transform([1, 2, 3], func=stateful_func, registry=registry) == [1, 4, 9]
    assert transformer.transform([1, 2, 3], func=stateful_func, registry=registry) == [4, 10, 18]


def test_sequence_transformer_transform_with_closure() -> None:
    multiplier = 3

    def multiply_by(x: Any) -> Any:
        return x * multiplier

    assert SequenceTransformer().transform(
        [1, 2, 3], func=multiply_by, registry=TransformerRegistry()
    ) == [3, 6, 9]
