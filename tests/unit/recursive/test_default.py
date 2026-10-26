from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never

import pytest

from coola.equality import objects_are_equal
from coola.recursive import DefaultTransformer, TransformerRegistry

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def registry() -> TransformerRegistry:
    return TransformerRegistry({object: DefaultTransformer()})


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
    data: Any, func: Callable, expected: Any, registry: TransformerRegistry
) -> None:
    assert objects_are_equal(registry.transform(data, func), expected)


def test_default_transformer_transform_with_dict_data(registry: TransformerRegistry) -> None:
    assert objects_are_equal(
        registry.transform({"a": 1, "b": 2}, lambda x: list(x.keys())),
        ["a", "b"],
    )


def test_default_transformer_transform_with_custom_function(registry: TransformerRegistry) -> None:
    def custom_func(x: Any) -> str:
        return f"processed: {x}"

    assert objects_are_equal(
        registry.transform("data", custom_func),
        "processed: data",
    )


def test_default_transformer_transform_with_none_data(registry: TransformerRegistry) -> None:
    assert objects_are_equal(registry.transform(None, lambda x: x), None)


def test_default_transformer_transform_with_identity_function(
    registry: TransformerRegistry,
) -> None:
    data = {"key": "value"}
    result = registry.transform(data, lambda x: x)
    assert result is data


def test_default_transformer_transform_with_exception_in_function(
    registry: TransformerRegistry,
) -> None:
    def failing_func(x: Any) -> Never:
        msg = f"Test error {x}"
        raise ValueError(msg)

    with pytest.raises(ValueError, match=r"Test error"):
        registry.transform(42, failing_func)


def test_default_transformer_transform_with_complex_data(registry: TransformerRegistry) -> None:
    assert objects_are_equal(
        registry.transform({"a": [1, 2], "b": {"c": 3}}, str),
        "{'a': [1, 2], 'b': {'c': 3}}",
    )


def test_default_transformer_transform_preserves_function_behavior(
    registry: TransformerRegistry,
) -> None:
    def increment_list(x: list) -> list:
        return [i + 1 for i in x]

    assert objects_are_equal(
        registry.transform([1, 2, 3], increment_list),
        [2, 3, 4],
    )


def _make_stateful_func() -> Callable[[Any], Any]:
    counter = {"count": 0}

    def stateful_func(x: Any) -> Any:
        counter["count"] += 1
        return x * counter["count"]

    return stateful_func


def test_default_transformer_transform_with_stateful_function(
    registry: TransformerRegistry,
) -> None:
    assert registry.transform(5, _make_stateful_func()) == 5


def test_default_transformer_transform_with_stateful_function_second_call(
    registry: TransformerRegistry,
) -> None:
    stateful_func = _make_stateful_func()
    registry.transform(5, stateful_func)
    assert registry.transform(5, stateful_func) == 10


def test_default_transformer_transform_with_closure(registry: TransformerRegistry) -> None:
    multiplier = 3

    def multiply_by(x: Any) -> Any:
        return x * multiplier

    assert registry.transform(7, multiply_by) == 21


def test_default_transformer_transform_with_closure_other_value(
    registry: TransformerRegistry,
) -> None:
    multiplier = 3

    def multiply_by(x: Any) -> Any:
        return x * multiplier

    assert registry.transform(1, multiply_by) == 3


@pytest.mark.parametrize("data", [1, "abc", [1, 2], {"a": 1}, None])
def test_default_transformer_expand_is_a_leaf(data: Any) -> None:
    assert DefaultTransformer().expand(data) is None
