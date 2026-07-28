from __future__ import annotations

from dataclasses import dataclass

from coola.nested import (
    convert_to_dict_of_lists,
    convert_to_json,
    convert_to_list_of_dicts,
)
from coola.testing.fixtures import numpy_available, pydantic_available, torch_available
from coola.utils.imports import (
    is_numpy_available,
    is_pydantic_available,
    is_torch_available,
)

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


##########################################
#     Tests for convert_to_json      #
##########################################


def test_convert_to_json_int() -> None:
    assert convert_to_json(1) == 1


def test_convert_to_json_dataclass() -> None:
    assert convert_to_json(Point(x=1, y=2)) == {"x": 1, "y": 2}


def test_convert_to_json_list_of_dataclasses() -> None:
    """Unlike to_json, convert_to_json recursively converts objects
    nested inside containers."""
    assert convert_to_json([Point(x=1, y=2), Point(x=3, y=4)]) == [
        {"x": 1, "y": 2},
        {"x": 3, "y": 4},
    ]


def test_convert_to_json_dict_of_dataclasses() -> None:
    assert convert_to_json({"a": Point(x=1, y=2), "b": Point(x=3, y=4)}) == {
        "a": {"x": 1, "y": 2},
        "b": {"x": 3, "y": 4},
    }


def test_convert_to_json_nested_containers() -> None:
    assert convert_to_json({"points": [Point(x=1, y=2), Point(x=3, y=4)]}) == {
        "points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
    }


def test_convert_to_json_no_conversion_needed() -> None:
    assert convert_to_json({"key": [1, 2, 3], "value": "abc"}) == {
        "key": [1, 2, 3],
        "value": "abc",
    }


@pydantic_available
def test_convert_to_json_list_of_pydantic_models() -> None:
    assert convert_to_json([MyModel(name="alice", age=30), MyModel(name="bob", age=25)]) == [
        {"name": "alice", "age": 30},
        {"name": "bob", "age": 25},
    ]


@numpy_available
def test_convert_to_json_numpy_array() -> None:
    assert convert_to_json(np.array([1, 2, 3])) == [1, 2, 3]


@numpy_available
def test_convert_to_json_dict_of_numpy_arrays() -> None:
    assert convert_to_json({"a": np.array([1, 2]), "b": np.array([3, 4])}) == {
        "a": [1, 2],
        "b": [3, 4],
    }


@torch_available
def test_convert_to_json_torch_tensor() -> None:
    assert convert_to_json(torch.tensor([1, 2, 3])) == [1, 2, 3]


@torch_available
def test_convert_to_json_dict_of_torch_tensors() -> None:
    assert convert_to_json({"a": torch.tensor([1, 2]), "b": torch.tensor([3, 4])}) == {
        "a": [1, 2],
        "b": [3, 4],
    }
