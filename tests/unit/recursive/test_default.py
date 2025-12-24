from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

import pytest

from coola import objects_are_equal
from coola.recursive import DefaultTransformer, TransformerRegistry

if TYPE_CHECKING:
    from collections.abc import Callable


def test_default_transformer_repr() -> None:
    assert repr(DefaultTransformer()) == "DefaultTransformer()"


def test_default_transformer_str() -> None:
    assert str(DefaultTransformer()) == "DefaultTransformer()"


@pytest.mark.parametrize(
    ("data", "func", "expected"),
    [
        # Basic type transformations
        pytest.param([1, 2, 3], str, "[1, 2, 3]", id="list_to_str"),
        pytest.param(42, lambda x: x * 2, 84, id="multiply_int"),
        pytest.param("hello", str.upper, "HELLO", id="str_upper"),
        pytest.param([1, 2, 3], len, 3, id="list_length"),
        pytest.param(5, lambda x: x**2, 25, id="square_int"),
        pytest.param("123", int, 123, id="str_to_int"),
        pytest.param(True, lambda x: not x, False, id="negate_bool"),
        pytest.param(3.14, round, 3, id="round_float"),
        # String operations
        pytest.param("lowercase", str.upper, "LOWERCASE", id="upper_case"),
        pytest.param("UPPERCASE", str.lower, "uppercase", id="lower_case"),
        pytest.param("  spaces  ", str.strip, "spaces", id="strip_spaces"),
        # Numeric operations
        pytest.param(10, lambda x: x + 5, 15, id="add_numbers"),
        pytest.param(100, lambda x: x / 10, 10.0, id="divide_numbers"),
        pytest.param(-5, abs, 5, id="absolute_value"),
        # Collection operations
        pytest.param([1, 2, 3, 4], lambda x: x[:2], [1, 2], id="slice_list"),
        pytest.param((1, 2, 3), list, [1, 2, 3], id="tuple_to_list"),
        pytest.param([3, 1, 2], sorted, [1, 2, 3], id="sort_list"),
        # Type conversions
        pytest.param(42, str, "42", id="int_to_string"),
        pytest.param("3.14", float, 3.14, id="string_to_float"),
        pytest.param([1, 2, 3], tuple, (1, 2, 3), id="list_to_tuple"),
    ],
)
def test_default_transformer_transform_parametrized(
    data: Any, func: Callable, expected: Any
) -> None:
    assert objects_are_equal(
        DefaultTransformer().transform(data, func=func, registry=TransformerRegistry()), expected
    )


def test_default_transformer_transform_with_dict_data() -> None:
    assert objects_are_equal(
        DefaultTransformer().transform(
            {"a": 1, "b": 2}, func=lambda x: list(x.keys()), registry=TransformerRegistry()
        ),
        ["a", "b"],
    )


def test_default_transformer_transform_with_custom_function() -> None:
    def custom_func(x: Any) -> str:
        return f"processed: {x}"

    assert objects_are_equal(
        DefaultTransformer().transform("data", func=custom_func, registry=TransformerRegistry()),
        "processed: data",
    )


def test_default_transformer_transform_with_none_data() -> None:
    assert objects_are_equal(
        DefaultTransformer().transform(None, func=lambda x: x, registry=TransformerRegistry()), None
    )


def test_default_transformer_transform_with_identity_function() -> None:
    data = {"key": "value"}
    result = DefaultTransformer().transform(data, func=lambda x: x, registry=TransformerRegistry())
    assert result is data


def test_default_transformer_transform_with_exception_in_function() -> None:
    def failing_func(x: Any) -> NoReturn:
        msg = f"Test error {x}"
        raise ValueError(msg)

    transformer = DefaultTransformer()
    registry = TransformerRegistry()
    with pytest.raises(ValueError, match="Test error"):
        transformer.transform(42, func=failing_func, registry=registry)


def test_default_transformer_transform_with_complex_data() -> None:
    assert objects_are_equal(
        DefaultTransformer().transform(
            {"a": [1, 2], "b": {"c": 3}}, func=str, registry=TransformerRegistry()
        ),
        "{'a': [1, 2], 'b': {'c': 3}}",
    )


def test_default_transformer_transform_preserves_function_behavior() -> None:
    def increment_list(x: list) -> list:
        return [i + 1 for i in x]

    assert objects_are_equal(
        DefaultTransformer().transform(
            [1, 2, 3], func=increment_list, registry=TransformerRegistry()
        ),
        [2, 3, 4],
    )


def test_default_transformer_transform_with_stateful_function() -> None:
    counter = {"count": 0}

    def stateful_func(x: Any) -> Any:
        counter["count"] += 1
        return x * counter["count"]

    transformer = DefaultTransformer()
    registry = TransformerRegistry()
    assert transformer.transform(5, func=stateful_func, registry=registry) == 5  # 5 * 1
    assert transformer.transform(5, func=stateful_func, registry=registry) == 10  # 5 * 2


def test_default_transformer_transform_with_closure() -> None:
    multiplier = 3

    def multiply_by(x: Any) -> Any:
        return x * multiplier

    transformer = DefaultTransformer()
    registry = TransformerRegistry()
    assert transformer.transform(7, func=multiply_by, registry=registry) == 21
    assert transformer.transform(1, func=multiply_by, registry=registry) == 3
