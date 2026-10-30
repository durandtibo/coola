from __future__ import annotations

from unittest.mock import patch

import pytest

from coola.jsonable import to_jsonable
from coola.jsonable.numpy import get_default_transformers
from coola.testing.marks import numpy_available
from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np


class Obj:
    pass


@numpy_available
def test_get_default_transformers_keys() -> None:
    assert list(get_default_transformers()) == [np.ndarray, np.generic]


def test_get_default_transformers_without_numpy() -> None:
    with patch("coola.jsonable.numpy.is_numpy_available", lambda: False):
        assert get_default_transformers() == {}


@numpy_available
def test_to_jsonable_array() -> None:
    assert to_jsonable(np.arange(3)) == [0, 1, 2]


@numpy_available
def test_to_jsonable_array_2d() -> None:
    assert to_jsonable(np.ones((2, 2))) == [[1.0, 1.0], [1.0, 1.0]]


@numpy_available
def test_to_jsonable_array_0d() -> None:
    assert to_jsonable(np.array(1.5)) == 1.5


@numpy_available
def test_to_jsonable_array_empty() -> None:
    assert to_jsonable(np.array([])) == []


@numpy_available
def test_to_jsonable_array_in_a_container() -> None:
    assert to_jsonable({"a": (np.arange(2),)}) == {"a": [[0, 1]]}


@numpy_available
@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(np.float64(1.5), 1.5, id="float64"),
        pytest.param(np.float32(1.5), 1.5, id="float32"),
        pytest.param(np.int64(2), 2, id="int64"),
        pytest.param(np.bool_(True), True, id="bool"),
        pytest.param(np.bytes_(b"a"), "YQ==", id="bytes"),
        pytest.param(np.datetime64("2020-01-02"), "2020-01-02", id="datetime64_day"),
        pytest.param(np.datetime64("NaT", "D"), None, id="nat"),
        pytest.param(np.timedelta64(5, "s"), 5.0, id="timedelta64"),
    ],
)
def test_to_jsonable_scalar(data: object, expected: object) -> None:
    assert to_jsonable(data) == expected


@numpy_available
def test_to_jsonable_float64_is_a_python_float() -> None:
    assert type(to_jsonable(np.float64(1.5))) is float


@numpy_available
def test_to_jsonable_datetime64_day_array() -> None:
    assert to_jsonable(np.array(["2020-01-01"], dtype="datetime64[D]")) == ["2020-01-01"]


@numpy_available
def test_to_jsonable_datetime64_nanosecond_is_an_int() -> None:
    assert to_jsonable(np.datetime64("2020-01-01T00:00:00.000000001")) == 1577836800000000001


@numpy_available
def test_to_jsonable_complex_array_raises() -> None:
    with pytest.raises(TypeError, match=r"Cannot convert an object of type complex"):
        to_jsonable(np.array([1 + 2j]))


@numpy_available
def test_to_jsonable_object_array_raises() -> None:
    with pytest.raises(TypeError, match=r"Cannot convert an object of type Obj"):
        to_jsonable(np.array([Obj()], dtype=object))


@numpy_available
def test_to_jsonable_complex_array_keep() -> None:
    assert to_jsonable(np.array([1 + 2j]), on_unsupported="keep") == [1 + 2j]
