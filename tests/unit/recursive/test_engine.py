from __future__ import annotations

from dataclasses import dataclass
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
from coola.recursive.engine import _check_layout, _zip, transform, transform_zip
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


@dataclass
class Point:
    x: int
    y: int


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

    with pytest.raises(RuntimeError, match="1"):
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
    with pytest.raises(TypeNotRegisteredError, match="is not registered"):
        transform(1, str, TransformerRegistry())


def test_recursive_apply_deep_structure() -> None:
    result = recursive_apply(nest(1, DEEP_NESTING_DEPTH), str)
    for _ in range(DEEP_NESTING_DEPTH):
        result = result[0]
    assert result == "1"


###################################
#     Tests for transform_zip     #
###################################


def zip_sum(items: Any) -> Any:
    return transform_zip(items, sum, get_default_registry())


def test_transform_zip_leaves() -> None:
    assert zip_sum([1, 2, 3]) == 6


def test_transform_zip_single_item() -> None:
    assert transform_zip([[1, 2]], lambda x: x[0] * 10, get_default_registry()) == [10, 20]


def test_transform_zip_nested() -> None:
    assert zip_sum([{"a": 1, "b": [2, 3]}, {"a": 10, "b": [20, 30]}]) == {"a": 11, "b": [22, 33]}


def test_transform_zip_empty_containers() -> None:
    assert zip_sum([{}, {}]) == {}


def test_transform_zip_func_receives_a_list_of_leaves() -> None:
    assert transform_zip([[1], [2]], lambda x: x, get_default_registry()) == [[1, 2]]


def test_transform_zip_keeps_the_type_of_the_first_item() -> None:
    assert zip_sum([(1, 2), [3, 4]]) == (4, 6)


def test_transform_zip_mapping_other_key_order() -> None:
    assert zip_sum([{"a": 1, "b": 2}, {"b": 20, "a": 10}]) == {"a": 11, "b": 22}


def test_transform_zip_mapping_other_key_order_nested() -> None:
    assert transform_zip(
        [{"a": 1, "b": 2}, {"b": 20, "a": 10}], lambda x: x, get_default_registry()
    ) == {"a": [1, 10], "b": [2, 20]}


def test_transform_zip_dataclass() -> None:
    assert zip_sum([Point(1, 2), Point(10, 20)]) == Point(11, 22)


def test_transform_zip_empty_items_raises() -> None:
    with pytest.raises(ValueError, match="items must contain at least one structure"):
        zip_sum([])


def test_transform_zip_different_keys_raises() -> None:
    with pytest.raises(
        ValueError,
        match=r"The mappings must have the same keys but item 0 has keys \['a'\] and item 1 has keys \['b'\]",
    ):
        zip_sum([{"a": 1}, {"b": 1}])


def test_transform_zip_different_lengths_raises() -> None:
    with pytest.raises(ValueError, match="same length but item 0 has 1 elements and item 1 has 2"):
        zip_sum([[1], [1, 2]])


def test_transform_zip_mapping_and_sequence_raises() -> None:
    with pytest.raises(ValueError, match="must have the same kind"):
        zip_sum([{"a": 1}, [1]])


def test_transform_zip_sequence_and_mapping_raises() -> None:
    with pytest.raises(ValueError, match="must have the same kind"):
        zip_sum([[1], {"a": 1}])


def test_transform_zip_leaf_and_container_raises() -> None:
    with pytest.raises(ValueError, match=r"item 0 is a leaf of type .* but item 1 is a container"):
        zip_sum([1, [1]])


def test_transform_zip_container_and_leaf_raises() -> None:
    with pytest.raises(ValueError, match=r"item 0 is a container of type .* but item 1 is a leaf"):
        zip_sum([[1], 1])


def test_transform_zip_sets_raises() -> None:
    with pytest.raises(TypeError, match="Sets cannot be combined"):
        zip_sum([{1}, {2}])


def test_transform_zip_cycle_raises() -> None:
    data: list[Any] = []
    data.append(data)
    with pytest.raises(ValueError, match="Cycle detected"):
        zip_sum([data, data])


def test_transform_zip_shared_container_is_not_a_cycle() -> None:
    shared = [1]
    assert zip_sum([[shared, shared], [[2], [3]]]) == [[3], [4]]


def test_transform_zip_deep_structure_raises_explanatory_error() -> None:
    with pytest.raises(RecursionError, match="recursion limit"):
        zip_sum([nest(1, DEEP_NESTING_DEPTH)] * 2)


def test_transform_zip_does_not_modify_the_items() -> None:
    items = [{"a": [1]}, {"a": [2]}]
    zip_sum(items)
    assert items == [{"a": [1]}, {"a": [2]}]


def test_zip_leaves() -> None:
    assert _zip([1, 2], sum, get_default_registry(), set()) == 3


def test_zip_releases_the_ancestors() -> None:
    ancestors: set[int] = set()
    _zip([[1], [2]], sum, get_default_registry(), ancestors)
    assert ancestors == set()


def test_check_layout_same_lengths() -> None:
    items = [[1], [2]]
    assert _check_layout(items, [(x, None) for x in items]) is None


def test_check_layout_different_lengths_raises() -> None:
    items = [[1], [2, 3]]
    with pytest.raises(ValueError, match="same length"):
        _check_layout(items, [(x, None) for x in items])
