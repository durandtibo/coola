from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from coola.nested import (
    convert_to_dict_of_lists,
    convert_to_jsonable,
    convert_to_list_of_dicts,
)
from coola.nested.conversion import _SetToListTransformer, _try_sorted_list
from coola.recursive import recursive_apply
from coola.testing.marks import numpy_available, pydantic_available, torch_available
from coola.utils.imports import (
    is_numpy_available,
    is_pydantic_available,
    is_torch_available,
)
from tests.helpers import nest

if is_pydantic_available():
    from pydantic import BaseModel
else:
    BaseModel = object

if is_numpy_available():
    import numpy as np

if is_torch_available():
    import torch


class MyModel(BaseModel):
    name: str
    age: int


@dataclass
class Point:
    x: int
    y: int


##############################################
#     Tests for convert_to_dict_of_lists     #
##############################################


def test_convert_to_dict_of_lists_empty_list() -> None:
    """Test convert_to_dict_of_lists with empty list returns empty
    dict."""
    assert convert_to_dict_of_lists([]) == {}


def test_convert_to_dict_of_lists_empty_dict() -> None:
    """Test convert_to_dict_of_lists with list containing empty dict."""
    assert convert_to_dict_of_lists([{}]) == {}


def test_convert_to_dict_of_lists() -> None:
    """Test convert_to_dict_of_lists with standard list of dicts."""
    assert convert_to_dict_of_lists(
        [{"key1": 1, "key2": 10}, {"key1": 2, "key2": 20}, {"key1": 3, "key2": 30}]
    ) == {
        "key1": [1, 2, 3],
        "key2": [10, 20, 30],
    }


def test_convert_to_dict_of_lists_single_item() -> None:
    """Test convert_to_dict_of_lists with single-item list."""
    assert convert_to_dict_of_lists([{"key1": 1, "key2": 2}]) == {
        "key1": [1],
        "key2": [2],
    }


def test_convert_to_dict_of_lists_different_types() -> None:
    """Test convert_to_dict_of_lists with different value types."""
    assert convert_to_dict_of_lists([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]) == {
        "name": ["Alice", "Bob"],
        "age": [30, 25],
    }


def test_convert_to_dict_of_lists_missing_key_raises_error() -> None:
    """Test convert_to_dict_of_lists raises when a later mapping is
    missing a key present in the first mapping."""
    with pytest.raises(ValueError, match="All the mappings must have the same keys"):
        convert_to_dict_of_lists([{"key1": 1, "key2": 10}, {"key1": 2}])


def test_convert_to_dict_of_lists_extra_key_raises_error() -> None:
    """Test convert_to_dict_of_lists raises when a later mapping has a
    key not present in the first mapping."""
    with pytest.raises(ValueError, match="All the mappings must have the same keys"):
        convert_to_dict_of_lists([{"key1": 1}, {"key1": 2, "key2": 20}])


##############################################
#     Tests for convert_to_list_of_dicts     #
##############################################


def test_convert_to_list_of_dicts_empty_dict() -> None:
    """Test convert_to_list_of_dicts with empty dict returns empty
    list."""
    assert convert_to_list_of_dicts({}) == []


def test_convert_to_list_of_dicts_empty_list() -> None:
    """Test convert_to_list_of_dicts with empty lists for all keys."""
    assert convert_to_list_of_dicts({"key1": [], "key2": []}) == []


def test_convert_to_list_of_dicts() -> None:
    """Test convert_to_list_of_dicts with standard dict of lists."""
    assert convert_to_list_of_dicts({"key1": [1, 2, 3], "key2": [10, 20, 30]}) == [
        {"key1": 1, "key2": 10},
        {"key1": 2, "key2": 20},
        {"key1": 3, "key2": 30},
    ]


def test_convert_to_list_of_dicts_single_value() -> None:
    """Test convert_to_list_of_dicts with single-value lists."""
    assert convert_to_list_of_dicts({"key1": [1], "key2": [2]}) == [{"key1": 1, "key2": 2}]


def test_convert_to_list_of_dicts_different_types() -> None:
    """Test convert_to_list_of_dicts with different value types."""
    assert convert_to_list_of_dicts({"name": ["Alice", "Bob"], "age": [30, 25]}) == [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ]


def test_convert_to_list_of_dicts_mismatched_length_raises_error() -> None:
    """Test convert_to_list_of_dicts raises when sequences have
    different lengths."""
    with pytest.raises(ValueError, match="All the sequences must have the same length"):
        convert_to_list_of_dicts({"key1": [1, 2, 3], "key2": [10, 20]})


#########################################
#     Tests for convert_to_jsonable     #
#########################################


def test_convert_to_json_int() -> None:
    assert convert_to_jsonable(1) == 1


def test_convert_to_json_dataclass() -> None:
    assert convert_to_jsonable(Point(x=1, y=2)) == {"x": 1, "y": 2}


def test_convert_to_json_list_of_dataclasses() -> None:
    """Unlike to_jsonable, convert_to_jsonable recursively converts
    objects nested inside containers."""
    assert convert_to_jsonable([Point(x=1, y=2), Point(x=3, y=4)]) == [
        {"x": 1, "y": 2},
        {"x": 3, "y": 4},
    ]


def test_convert_to_json_dict_of_dataclasses() -> None:
    assert convert_to_jsonable({"a": Point(x=1, y=2), "b": Point(x=3, y=4)}) == {
        "a": {"x": 1, "y": 2},
        "b": {"x": 3, "y": 4},
    }


def test_convert_to_json_nested_containers() -> None:
    assert convert_to_jsonable({"points": [Point(x=1, y=2), Point(x=3, y=4)]}) == {
        "points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    }


def test_convert_to_json_no_conversion_needed() -> None:
    assert convert_to_jsonable({"key": [1, 2, 3], "value": "abc"}) == {
        "key": [1, 2, 3],
        "value": "abc",
    }


@pydantic_available
def test_convert_to_json_list_of_pydantic_models() -> None:
    assert convert_to_jsonable([MyModel(name="alice", age=30), MyModel(name="bob", age=25)]) == [
        {"name": "alice", "age": 30},
        {"name": "bob", "age": 25},
    ]


@numpy_available
def test_convert_to_json_numpy_array() -> None:
    assert convert_to_jsonable(np.array([1, 2, 3])) == [1, 2, 3]


@numpy_available
def test_convert_to_json_dict_of_numpy_arrays() -> None:
    assert convert_to_jsonable({"a": np.array([1, 2]), "b": np.array([3, 4])}) == {
        "a": [1, 2],
        "b": [3, 4],
    }


@torch_available
def test_convert_to_json_torch_tensor() -> None:
    assert convert_to_jsonable(torch.tensor([1, 2, 3])) == [1, 2, 3]


@torch_available
def test_convert_to_json_dict_of_torch_tensors() -> None:
    assert convert_to_jsonable({"a": torch.tensor([1, 2]), "b": torch.tensor([3, 4])}) == {
        "a": [1, 2],
        "b": [3, 4],
    }


def test_convert_to_jsonable_set_and_range() -> None:
    data = {"a": {3, 1, 2}, "b": [frozenset({"x"})], "c": range(2)}
    out = convert_to_jsonable(data)
    assert out == {"a": [1, 2, 3], "b": [["x"]], "c": [0, 1]}
    json.dumps(out)


def test_try_sorted_list_set() -> None:
    assert _try_sorted_list({3, 1, 2}) == [1, 2, 3]


def test_try_sorted_list_frozenset() -> None:
    assert _try_sorted_list(frozenset({"b", "a"})) == ["a", "b"]


def test_try_sorted_list_not_comparable() -> None:
    assert sorted(_try_sorted_list({1, "a"}), key=str) == [1, "a"]


@pytest.mark.parametrize("value", ["ab", b"ab", bytearray(b"ab")])
def test_convert_to_list_of_dicts_string_value_raises(value: object) -> None:
    with pytest.raises(TypeError, match="must be a sequence of items"):
        convert_to_list_of_dicts({"a": value, "b": [1, 2]})


def test_convert_to_jsonable_deep_structure() -> None:
    result = convert_to_jsonable(nest(1, 5000))
    for _ in range(5000):
        result = result[0]
    assert result == 1


def test_convert_to_jsonable_list_that_contains_itself() -> None:
    data: list = []
    data.append(data)
    with pytest.raises(ValueError, match="Cycle detected"):
        convert_to_jsonable(data)


def test_set_to_list_transformer_expand_children() -> None:
    children, _ = _SetToListTransformer().expand({1})
    assert children == [1]


def test_set_to_list_transformer_expand_rebuild_sorts_the_items() -> None:
    _, rebuild = _SetToListTransformer().expand({3, 1, 2})
    assert rebuild([3, 1, 2]) == [1, 2, 3]


def test_set_to_list_transformer_repr() -> None:
    assert repr(_SetToListTransformer()) == "_SetToListTransformer()"


def test_convert_to_jsonable_set_in_a_set_of_frozensets() -> None:
    assert convert_to_jsonable({frozenset({2, 1})}) == [[1, 2]]


def test_convert_to_jsonable_does_not_modify_the_default_registry() -> None:
    convert_to_jsonable({1})
    assert recursive_apply({1}, str) == {"1"}
