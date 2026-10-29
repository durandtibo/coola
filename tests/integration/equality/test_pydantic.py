from __future__ import annotations

import pytest

from coola.equality import objects_are_allclose, objects_are_equal
from coola.testing.marks import numpy_available, pydantic_available
from coola.utils.imports import is_numpy_available, is_pydantic_available

if is_numpy_available():
    import numpy as np

if is_pydantic_available():
    from pydantic import BaseModel, ConfigDict

    class Values(BaseModel):
        x: list[float]

    class WithArray(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        x: np.ndarray


class Raising:
    def __eq__(self, other: object) -> bool:
        msg = "ambiguous"
        raise ValueError(msg)

    def __hash__(self) -> int:
        return 0


@pydantic_available
@numpy_available
def test_objects_are_equal_pydantic_model_with_array_equal() -> None:
    assert objects_are_equal(WithArray(x=np.ones(2)), WithArray(x=np.ones(2)))


@pydantic_available
@numpy_available
def test_objects_are_equal_pydantic_model_with_array_different() -> None:
    assert not objects_are_equal(WithArray(x=np.ones(2)), WithArray(x=np.zeros(2)))


@pydantic_available
def test_objects_are_allclose_pydantic_model_within_tolerance() -> None:
    assert objects_are_allclose(Values(x=[1.0]), Values(x=[1.0 + 1e-9]))


@pydantic_available
def test_objects_are_equal_pydantic_model_equal_nan() -> None:
    assert objects_are_equal(Values(x=[float("nan")]), Values(x=[float("nan")]), equal_nan=True)


@pydantic_available
def test_objects_are_equal_pydantic_model_nested_in_a_dict() -> None:
    assert objects_are_equal({"a": Values(x=[1.0])}, {"a": Values(x=[1.0])})


def test_objects_are_equal_eq_raising_is_reported_as_type_error() -> None:
    with pytest.raises(TypeError, match="Register an equality tester"):
        objects_are_equal(Raising(), Raising())
