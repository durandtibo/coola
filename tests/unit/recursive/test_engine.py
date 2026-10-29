from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from coola.recursive import (
    BaseTransformer,
    DefaultTransformer,
    IdentityTransformer,
    SequenceTransformer,
    TransformerRegistry,
    get_default_registry,
    recursive_apply,
)
from coola.recursive.engine import transform
from coola.registry import TypeNotRegisteredError
from tests.helpers import DEEP_NESTING_DEPTH

if TYPE_CHECKING:
    from coola.recursive.base import Expansion


class SelfChildTransformer(BaseTransformer[list]):
    r"""A custom transformer that returns its own object as a child."""

    def expand(self, data: list) -> Expansion | None:
        return [data], lambda items: items


@pytest.fixture
def registry() -> TransformerRegistry:
    return get_default_registry()


def nest(value: Any, depth: int) -> Any:
    for _ in range(depth):
        value = [value]
    return value


def test_transform_leaf(registry: TransformerRegistry) -> None:
    assert transform(1, str, registry) == "1"


def test_transform_order_of_the_calls(registry: TransformerRegistry) -> None:
    calls = []

    def record(x: Any) -> Any:
        calls.append(x)
        return x

    transform([1, {"a": 2, "b": (3, [4])}, 5], record, registry)
    assert calls == [1, 2, 3, 4, 5]


def test_transform_func_called_once_per_occurrence(registry: TransformerRegistry) -> None:
    calls = []
    shared = [1]

    def record(x: Any) -> Any:
        calls.append(x)
        return x

    transform([shared, shared], record, registry)
    assert calls == [1, 1]


def test_transform_output_of_a_shared_container_has_no_aliasing(
    registry: TransformerRegistry,
) -> None:
    shared = [1]
    result = transform([shared, shared], str, registry)
    assert result[0] is not result[1]


def test_transform_does_not_modify_the_input(registry: TransformerRegistry) -> None:
    data = {"a": [1, 2], "b": (3,)}
    transform(data, str, registry)
    assert data == {"a": [1, 2], "b": (3,)}


def test_transform_whole_output(registry: TransformerRegistry) -> None:
    result = transform({"a": [1, (2, 3)], "b": {4}, "c": "x", "d": {}}, str, registry)
    assert result == {"a": ["1", ("2", "3")], "b": {"4"}, "c": "x", "d": {}}


def test_transform_list_that_contains_itself(registry: TransformerRegistry) -> None:
    data: list[Any] = []
    data.append(data)
    with pytest.raises(ValueError, match="Cycle detected: an object of type <class 'list'>"):
        transform(data, str, registry)


def test_transform_dict_that_contains_itself(registry: TransformerRegistry) -> None:
    data: dict[str, Any] = {}
    data["a"] = data
    with pytest.raises(ValueError, match="Cycle detected: an object of type <class 'dict'>"):
        transform(data, str, registry)


def test_transform_tuple_in_a_list_that_contains_itself(registry: TransformerRegistry) -> None:
    data: list[Any] = []
    data.append((data,))
    with pytest.raises(ValueError, match="Cycle detected"):
        transform(data, str, registry)


def test_transform_cycle_with_a_custom_transformer() -> None:
    registry = TransformerRegistry({object: DefaultTransformer(), list: SelfChildTransformer()})
    with pytest.raises(ValueError, match="Cycle detected"):
        transform([], str, registry)


def test_transform_works_after_a_cycle(registry: TransformerRegistry) -> None:
    data: list[Any] = []
    data.append(data)
    with pytest.raises(ValueError, match="Cycle detected"):
        transform(data, str, registry)
    assert transform([1, [2]], str, registry) == ["1", ["2"]]


def test_transform_deep_structure(registry: TransformerRegistry) -> None:
    result = transform(nest(1, DEEP_NESTING_DEPTH), str, registry)
    for _ in range(DEEP_NESTING_DEPTH):
        result = result[0]
    assert result == "1"


def test_transform_exception_of_func_propagates(registry: TransformerRegistry) -> None:
    def failing(x: Any) -> Any:
        msg = f"failed on {x}"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="failed on 2"):
        transform([1, [2]], lambda x: x if x == 1 else failing(x), registry)


def test_transform_works_after_an_exception_of_func(registry: TransformerRegistry) -> None:
    def failing(x: Any) -> Any:
        raise RuntimeError(x)

    with pytest.raises(RuntimeError):
        transform([1, [2]], failing, registry)
    assert transform([1, [2]], str, registry) == ["1", ["2"]]


def test_transform_reentrant_call(registry: TransformerRegistry) -> None:
    def func(x: Any) -> Any:
        return transform([x, [x]], str, registry)

    assert transform([1], func, registry) == [["1", ["1"]]]


def test_transform_custom_registry_is_independent(registry: TransformerRegistry) -> None:
    custom = TransformerRegistry({object: DefaultTransformer()})
    assert transform([1], str, custom) == "[1]"
    assert transform([1], str, registry) == ["1"]


def test_transform_transformer_registered_during_a_call_is_not_used() -> None:
    registry = TransformerRegistry({object: DefaultTransformer(), list: SequenceTransformer()})

    def func(x: Any) -> Any:
        registry.register(int, IdentityTransformer(), exist_ok=True)
        return x * 10

    # The transformer of int is resolved before the first call of func and cached for this call.
    assert transform([1, 2], func, registry) == [10, 20]


def test_transform_empty_registry_raises() -> None:
    with pytest.raises(TypeNotRegisteredError):
        transform(1, str, TransformerRegistry())


def test_recursive_apply_deep_structure() -> None:
    result = recursive_apply(nest(1, DEEP_NESTING_DEPTH), str)
    for _ in range(DEEP_NESTING_DEPTH):
        result = result[0]
    assert result == "1"
