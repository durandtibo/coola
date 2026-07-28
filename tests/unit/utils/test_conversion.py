from __future__ import annotations

from dataclasses import dataclass

from coola.testing.fixtures import numpy_available, pydantic_available, torch_available
from coola.utils.conversion import to_json
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


##################################
#     Tests for to_json      #
##################################


def test_to_json_dataclass() -> None:
    assert to_json(Point(x=1, y=2)) == {"x": 1, "y": 2}


def test_to_json_nested_dataclass() -> None:
    assert to_json(Line(start=Point(x=1, y=2), end=Point(x=3, y=4))) == {
        "start": {"x": 1, "y": 2},
        "end": {"x": 3, "y": 4},
    }


def test_to_json_int() -> None:
    assert to_json(1) == 1


def test_to_json_str() -> None:
    assert to_json("abc") == "abc"


def test_to_json_none() -> None:
    assert to_json(None) is None


def test_to_json_list_not_converted() -> None:
    """A list is not a dataclass or BaseModel, so it is returned
    unchanged, even if it contains objects that would be converted."""
    data = [Point(x=1, y=2)]
    assert to_json(data) is data


def test_to_json_dict_not_converted() -> None:
    """A dict is not a dataclass or BaseModel, so it is returned
    unchanged, even if it contains objects that would be converted."""
    data = {"point": Point(x=1, y=2)}
    assert to_json(data) is data


@pydantic_available
def test_to_json_pydantic_model() -> None:
    assert to_json(MyModel(name="alice", age=30)) == {"name": "alice", "age": 30}


@numpy_available
def test_to_json_numpy_array() -> None:
    assert to_json(np.array([1, 2, 3])) == [1, 2, 3]


@numpy_available
def test_to_json_numpy_array_2d() -> None:
    assert to_json(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]


@torch_available
def test_to_json_torch_tensor() -> None:
    assert to_json(torch.tensor([1, 2, 3])) == [1, 2, 3]


@torch_available
def test_to_json_torch_tensor_2d() -> None:
    assert to_json(torch.tensor([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]
