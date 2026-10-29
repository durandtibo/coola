from __future__ import annotations

from collections import OrderedDict, defaultdict
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Never

import pytest

from coola.equality import objects_are_equal
from coola.recursive import DefaultTransformer, MappingTransformer, TransformerRegistry
from tests.unit.recursive.helpers import expand_not_none

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def registry() -> TransformerRegistry:
    transformer = MappingTransformer()
    return TransformerRegistry(
        {
            object: DefaultTransformer(),
            dict: transformer,
            OrderedDict: transformer,
            defaultdict: transformer,
        }
    )


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
    data: Any, func: Callable, expected: Any, registry: TransformerRegistry
) -> None:
    assert objects_are_equal(registry.transform(data, func), expected)


def test_mapping_transformer_transform_nested_dict(registry: TransformerRegistry) -> None:
    assert objects_are_equal(
        registry.transform({"a": 1, "b": {"x": 3, "y": 4}}, str),
        {"a": "1", "b": {"x": "3", "y": "4"}},
    )


def test_mapping_transformer_transform_with_custom_function(registry: TransformerRegistry) -> None:
    def custom_func(x: Any) -> str:
        return f"processed: {x}"

    assert objects_are_equal(
        registry.transform({"a": 1, "b": 2}, custom_func),
        {"a": "processed: 1", "b": "processed: 2"},
    )


def test_mapping_transformer_transform_with_identity_function(
    registry: TransformerRegistry,
) -> None:
    data = {"a": 1, "b": 2}
    result = registry.transform(data, lambda x: x)
    assert result is not data


def test_mapping_transformer_transform_with_exception_in_function(
    registry: TransformerRegistry,
) -> None:
    def failing_func(x: Any) -> Never:
        msg = f"Test error {x}"
        raise ValueError(msg)

    with pytest.raises(ValueError, match=r"Test error"):
        registry.transform({"a": 1, "b": 2}, failing_func)


def test_mapping_transformer_transform_preserves_function_behavior(
    registry: TransformerRegistry,
) -> None:
    def create_list(x: Any) -> list:
        return [x, x + 1, x + 2]

    assert objects_are_equal(
        registry.transform({"a": 1, "b": 2}, create_list),
        {"a": [1, 2, 3], "b": [2, 3, 4]},
    )


def _make_stateful_func() -> Callable[[Any], Any]:
    counter = {"count": 0}

    def stateful_func(x: Any) -> Any:
        counter["count"] += 1
        return x * counter["count"]

    return stateful_func


def test_mapping_transformer_transform_with_stateful_function(
    registry: TransformerRegistry,
) -> None:
    assert registry.transform({"a": 1, "b": 2}, _make_stateful_func()) == {"a": 1, "b": 4}


def test_mapping_transformer_transform_with_stateful_function_second_call(
    registry: TransformerRegistry,
) -> None:
    stateful_func = _make_stateful_func()
    registry.transform({"a": 1, "b": 2}, stateful_func)
    assert registry.transform({"a": 1, "b": 2}, stateful_func) == {
        "a": 3,
        "b": 8,
    }


def test_mapping_transformer_transform_with_closure(registry: TransformerRegistry) -> None:
    multiplier = 3

    def multiply_by(x: Any) -> Any:
        return x * multiplier

    assert registry.transform({"a": 1, "b": 2}, multiply_by) == {"a": 3, "b": 6}


def test_mapping_transformer_transform_defaultdict_type(registry: TransformerRegistry) -> None:
    result = registry.transform(defaultdict[str, Any](list, {"a": 1}), lambda x: x + 1)
    assert type(result) is defaultdict


def test_mapping_transformer_transform_defaultdict_default_factory(
    registry: TransformerRegistry,
) -> None:
    result = registry.transform(defaultdict[str, Any](list, {"a": 1}), lambda x: x + 1)
    assert result.default_factory is list


def test_mapping_transformer_transform_defaultdict_values(registry: TransformerRegistry) -> None:
    result = registry.transform(defaultdict[str, Any](list, {"a": 1}), lambda x: x + 1)
    assert result == {"a": 2}


def test_mapping_transformer_expand_children() -> None:
    children, _ = expand_not_none(MappingTransformer(), {"a": 1, "b": 2})
    assert children == [1, 2]


def test_mapping_transformer_expand_rebuild() -> None:
    _, rebuild = expand_not_none(MappingTransformer(), {"a": 1, "b": 2})
    assert rebuild(["x", "y"]) == {"a": "x", "b": "y"}


def test_mapping_transformer_expand_rebuild_keeps_the_order_of_the_keys() -> None:
    _, rebuild = expand_not_none(MappingTransformer(), {"b": 1, "a": 2})
    assert list(rebuild([3, 4])) == ["b", "a"]


def test_mapping_transformer_expand_rebuild_ordered_dict() -> None:
    _, rebuild = expand_not_none(MappingTransformer(), OrderedDict(a=1))
    assert type(rebuild([2])) is OrderedDict


def test_mapping_transformer_expand_rebuild_defaultdict_keeps_the_factory() -> None:
    _, rebuild = expand_not_none(MappingTransformer(), defaultdict(int, {"a": 1}))
    assert rebuild([2]).default_factory is int


def test_mapping_transformer_expand_rebuild_mapping_proxy() -> None:
    _, rebuild = expand_not_none(MappingTransformer(), MappingProxyType({"a": 1}))
    assert rebuild([2]) == MappingProxyType({"a": 2})


def test_mapping_transformer_expand_empty() -> None:
    children, rebuild = expand_not_none(MappingTransformer(), {})
    assert (children, rebuild([])) == ([], {})


def test_mapping_transformer_expand_does_not_modify_the_input() -> None:
    data = {"a": 1, "b": 2}
    MappingTransformer().expand(data)
    assert data == {"a": 1, "b": 2}
