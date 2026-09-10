from __future__ import annotations

from dataclasses import dataclass

from coola.testing.fixtures import numpy_available, pydantic_available, torch_available
from coola.utils.conversion import to_jsonable
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


@dataclass
class Line:
    start: Point
    end: Point


@dataclass
class Container:
    items: list
    mapping: dict


@dataclass
class Wrapper:
    value: object


##############################
#     Tests for to_jsonable      #
##############################


def test_to_json_dataclass() -> None:
    assert to_jsonable(Point(x=1, y=2)) == {"x": 1, "y": 2}


def test_to_json_nested_dataclass() -> None:
    assert to_jsonable(Line(start=Point(x=1, y=2), end=Point(x=3, y=4))) == {
        "start": {"x": 1, "y": 2},
        "end": {"x": 3, "y": 4},
    }


def test_to_json_int() -> None:
    assert to_jsonable(1) == 1


def test_to_json_str() -> None:
    assert to_jsonable("abc") == "abc"


def test_to_json_none() -> None:
    assert to_jsonable(None) is None


def test_to_json_list_not_converted() -> None:
    """A list is not a dataclass or BaseModel, so it is returned
    unchanged, even if it contains objects that would be converted."""
    data = [Point(x=1, y=2)]
    assert to_jsonable(data) is data


def test_to_json_dict_not_converted() -> None:
    """A dict is not a dataclass or BaseModel, so it is returned
    unchanged, even if it contains objects that would be converted."""
    data = {"point": Point(x=1, y=2)}
    assert to_jsonable(data) is data


@pydantic_available
def test_to_json_pydantic_model() -> None:
    assert to_jsonable(MyModel(name="alice", age=30)) == {"name": "alice", "age": 30}


@numpy_available
def test_to_json_numpy_array() -> None:
    assert to_jsonable(np.array([1, 2, 3])) == [1, 2, 3]


@numpy_available
def test_to_json_numpy_array_2d() -> None:
    assert to_jsonable(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]


@torch_available
def test_to_json_torch_tensor() -> None:
    assert to_jsonable(torch.tensor([1, 2, 3])) == [1, 2, 3]


@torch_available
def test_to_json_torch_tensor_2d() -> None:
    assert to_jsonable(torch.tensor([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]


@numpy_available
def test_to_json_dataclass_with_nested_numpy_array() -> None:
    """A numpy.ndarray field nested inside a dataclass must be
    converted, even though dataclasses.asdict does not convert it on its
    own."""
    assert to_jsonable(Wrapper(value=np.array([1, 2, 3]))) == {"value": [1, 2, 3]}


@torch_available
def test_to_json_dataclass_with_nested_torch_tensor() -> None:
    """A torch.Tensor field nested inside a dataclass must be converted,
    even though dataclasses.asdict does not convert it on its own."""
    assert to_jsonable(Wrapper(value=torch.tensor([1, 2, 3]))) == {"value": [1, 2, 3]}


@pydantic_available
def test_to_json_dataclass_with_nested_pydantic_model() -> None:
    """A pydantic.BaseModel field nested inside a dataclass must be
    converted, even though dataclasses.asdict does not convert it on its
    own."""
    assert to_jsonable(Wrapper(value=MyModel(name="alice", age=30))) == {
        "value": {"name": "alice", "age": 30}
    }


@numpy_available
def test_to_json_dataclass_with_nested_numpy_array_in_list() -> None:
    """A numpy.ndarray nested inside a list field of a dataclass must be
    converted."""
    assert to_jsonable(Container(items=[np.array([1, 2])], mapping={})) == {
        "items": [[1, 2]],
        "mapping": {},
    }


@numpy_available
def test_to_json_dataclass_with_nested_numpy_array_in_dict() -> None:
    """A numpy.ndarray nested inside a dict field of a dataclass must be
    converted."""
    assert to_jsonable(Container(items=[], mapping={"data": np.array([1, 2])})) == {
        "items": [],
        "mapping": {"data": [1, 2]},
    }
