from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath

import pytest

from coola.nested import (
    convert_to_dict_of_lists,
    convert_to_jsonable,
    convert_to_list_of_dicts,
)
from coola.nested.conversion import _SetToListTransformer, _sorted_elements
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


@dataclass
class Wrapper:
    value: object


@dataclass
class Shape:
    points: list
    tags: set
    corner: tuple


class Color(Enum):
    RED = 1


class NoCopy:
    def __deepcopy__(self, memo: dict) -> None:
        msg = "must not be copied"
        raise TypeError(msg)


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


#######################################################
#     Tests for convert_to_jsonable and dataclasses   #
#######################################################


def test_convert_to_jsonable_dataclass() -> None:
    assert convert_to_jsonable(Point(x=1, y=2)) == {"x": 1, "y": 2}


def test_convert_to_jsonable_nested_dataclass() -> None:
    assert convert_to_jsonable(Wrapper(value=Wrapper(value=Point(x=1, y=2)))) == {
        "value": {"value": {"x": 1, "y": 2}}
    }


def test_convert_to_jsonable_dataclass_whole_output() -> None:
    data = Shape(points=[Point(1, 2), Point(3, 4)], tags={"b", "a", "c"}, corner=(0, (1, 2)))
    assert convert_to_jsonable(data) == {
        "points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
        "tags": ["a", "b", "c"],
        "corner": (0, (1, 2)),
    }


def test_convert_to_jsonable_tuple_in_a_dataclass_stays_a_tuple() -> None:
    assert convert_to_jsonable(Wrapper(value=(1, 2))) == {"value": (1, 2)}


def test_convert_to_jsonable_tuple_outside_a_dataclass_stays_a_tuple() -> None:
    assert convert_to_jsonable({"value": (1, 2)}) == {"value": (1, 2)}


def test_convert_to_jsonable_set_in_a_dataclass_is_sorted() -> None:
    assert convert_to_jsonable(Wrapper(value={"d", "a", "b", "c", "e"})) == {
        "value": ["a", "b", "c", "d", "e"]
    }


def test_convert_to_jsonable_dataclass_in_a_set_in_a_dataclass() -> None:
    assert convert_to_jsonable(Wrapper(value={frozenset({2, 1})})) == {"value": [[1, 2]]}


def test_convert_to_jsonable_dataclass_in_a_list_and_in_a_dict() -> None:
    assert convert_to_jsonable([Point(1, 2), {"key": Point(3, 4)}]) == [
        {"x": 1, "y": 2},
        {"key": {"x": 3, "y": 4}},
    ]


def test_convert_to_jsonable_enum_in_a_dataclass() -> None:
    assert convert_to_jsonable(Wrapper(value=Color.RED)) == {"value": 1}


def test_convert_to_jsonable_path_in_a_dataclass() -> None:
    assert convert_to_jsonable(Wrapper(value=PurePosixPath("a/b"))) == {"value": "a/b"}


@numpy_available
def test_convert_to_jsonable_dataclass_with_numpy_array() -> None:
    assert convert_to_jsonable(Wrapper(value=np.array([1, 2, 3]))) == {"value": [1, 2, 3]}


@torch_available
def test_convert_to_jsonable_dataclass_with_torch_tensor() -> None:
    assert convert_to_jsonable(Wrapper(value=torch.tensor([1, 2, 3]))) == {"value": [1, 2, 3]}


@pydantic_available
def test_convert_to_jsonable_dataclass_with_pydantic_model() -> None:
    assert convert_to_jsonable(Wrapper(value=MyModel(name="alice", age=30))) == {
        "value": {"name": "alice", "age": 30}
    }


@numpy_available
def test_convert_to_jsonable_dataclass_with_numpy_array_in_a_list_and_in_a_dict() -> None:
    data = Wrapper(value=[np.array([1, 2]), {"data": np.array([3])}])
    assert convert_to_jsonable(data) == {"value": [[1, 2], {"data": [3]}]}


def test_convert_to_jsonable_dataclass_does_not_copy_the_fields() -> None:
    obj = NoCopy()
    assert convert_to_jsonable(Wrapper(value=obj))["value"] is obj


def test_convert_to_jsonable_dataclass_with_lock() -> None:
    assert set(convert_to_jsonable(Wrapper(value=threading.Lock()))) == {"value"}


def test_convert_to_jsonable_dataclass_does_not_modify_the_input() -> None:
    data = Wrapper(value={"b", "a"})
    convert_to_jsonable(data)
    assert data == Wrapper(value={"b", "a"})


def test_convert_to_jsonable_empty_dataclass() -> None:
    @dataclass
    class Empty:
        pass

    assert convert_to_jsonable({"a": Empty()}) == {"a": {}}


def test_convert_to_jsonable_deep_dataclass_does_not_raise() -> None:
    data = Point(1, 2)
    for _ in range(5000):
        data = Wrapper(value=data)
    result = convert_to_jsonable(data)
    for _ in range(5000):
        result = result["value"]
    assert result == {"x": 1, "y": 2}


def test_convert_to_jsonable_cycle_through_a_dataclass_raises() -> None:
    data = Wrapper(value=[])
    data.value.append(data)
    with pytest.raises(ValueError, match="Cycle detected"):
        convert_to_jsonable(data)


###################################
#     Tests for _sorted_elements   #
###################################


def test_sorted_elements_set() -> None:
    assert _sorted_elements({3, 1, 2}) == [1, 2, 3]


def test_sorted_elements_frozenset() -> None:
    assert _sorted_elements(frozenset({"b", "a"})) == ["a", "b"]


def test_sorted_elements_empty() -> None:
    assert _sorted_elements(set()) == []


def test_sorted_elements_mixed_types_sorted_by_type_name_then_repr() -> None:
    assert _sorted_elements({1, "a", 2, "b"}) == [1, 2, "a", "b"]


def test_sorted_elements_mixed_types_does_not_depend_on_the_order() -> None:
    assert _sorted_elements(["a", 1]) == _sorted_elements([1, "a"])


def test_sorted_elements_int_and_float_are_sorted_by_value() -> None:
    assert _sorted_elements({2.5, 1, 3}) == [1, 2.5, 3]


def test_sorted_elements_nan_does_not_depend_on_the_order() -> None:
    nan = float("nan")
    assert repr(_sorted_elements([nan, 1.0, 2.0])) == repr(_sorted_elements([2.0, nan, 1.0]))


def test_sorted_elements_nan_whole_output() -> None:
    assert repr(_sorted_elements([float("nan"), 2.0, 1.0])) == "[1.0, 2.0, nan]"


def test_sorted_elements_lists_of_different_types() -> None:
    assert _sorted_elements([["a"], [1]]) == [["a"], [1]]


def test_convert_to_jsonable_set_of_mixed_types() -> None:
    assert convert_to_jsonable({"a", 1, 2}) == [1, 2, "a"]
