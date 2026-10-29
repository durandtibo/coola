r"""Check that the default ``max_depth`` protects against the
interpreter recursion limit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from coola.equality import MaxDepthExceededError, objects_are_equal
from coola.testing.marks import pydantic_available
from coola.utils.imports import is_pydantic_available

if TYPE_CHECKING:
    from collections.abc import Callable

if is_pydantic_available():
    from pydantic import BaseModel

    class Model(BaseModel):
        child: Any = None


@dataclass
class Node:
    child: Any = None


def nest_dict(depth: int) -> dict:
    value: Any = 1
    for _ in range(depth):
        value = {"a": value}
    return value


def nest_list(depth: int) -> list:
    value: Any = 1
    for _ in range(depth):
        value = [value]
    return value


def nest_tuple(depth: int) -> tuple:
    value: Any = 1
    for _ in range(depth):
        value = (value,)
    return value


def nest_dataclass(depth: int) -> Node:
    value: Any = 1
    for _ in range(depth):
        value = Node(value)
    return value


def nest_model(depth: int) -> Model:
    value: Any = 1
    for _ in range(depth):
        value = Model(child=value)
    return value


CASES = [
    pytest.param(nest_dict, id="dict"),
    pytest.param(nest_list, id="list"),
    pytest.param(nest_tuple, id="tuple"),
    pytest.param(nest_dataclass, id="dataclass"),
    pytest.param(nest_model, id="pydantic", marks=pydantic_available),
]

# The caller (a test runner, a web framework) uses frames of the same recursion limit.
CALLER_FRAMES = 100


def call_from_a_deep_caller(frames: int, function: Callable[[], bool]) -> bool:
    if frames == 0:
        return function()
    return call_from_a_deep_caller(frames - 1, function)


@pytest.mark.parametrize("nest", CASES)
def test_objects_are_equal_default_max_depth_is_reached_before_the_interpreter_limit(
    nest: Callable[[int], object],
) -> None:
    """Test that the default ``max_depth`` of 100 is below the
    interpreter limit, from a caller that is 100 frames deep."""
    with pytest.raises(MaxDepthExceededError, match=r"Maximum recursion depth \(100\) exceeded"):
        call_from_a_deep_caller(CALLER_FRAMES, lambda: objects_are_equal(nest(102), nest(102)))


@pytest.mark.parametrize("nest", CASES)
def test_objects_are_equal_default_max_depth_accepts_100_levels(
    nest: Callable[[int], object],
) -> None:
    """Test that 100 nested levels are compared with the default
    ``max_depth``, from a caller that is 100 frames deep."""
    assert call_from_a_deep_caller(CALLER_FRAMES, lambda: objects_are_equal(nest(100), nest(100)))
