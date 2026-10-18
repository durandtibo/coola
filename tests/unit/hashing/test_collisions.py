from __future__ import annotations

from collections import OrderedDict, deque, namedtuple
from datetime import date, datetime, timezone
from enum import Enum, IntEnum, StrEnum
from pathlib import Path

import pytest

from coola.hashing import hash_object
from coola.testing.fixtures import numpy_available
from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np

Point = namedtuple("Point", ["x", "y"])
Other = namedtuple("Other", ["x", "y"])


class Color(Enum):
    RED = 1


class Number(IntEnum):
    ONE = 1


class Label(StrEnum):
    A = "a"


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(1, "1", id="int-str"),
        pytest.param(1.5, "1.5", id="float-str"),
        pytest.param(True, "True", id="bool-str"),
        pytest.param(None, "None", id="none-str"),
        pytest.param([], "", id="empty-list-str"),
        pytest.param({}, "", id="empty-dict-str"),
        pytest.param([], {}, id="empty-list-dict"),
        pytest.param(b"", "", id="empty-bytes-str"),
        pytest.param(b"abc", "abc", id="bytes-str"),
        pytest.param(["a"], "a", id="list-str"),
    ],
)
def test_hash_object_different_types_do_not_collide(first: object, second: object) -> None:
    assert hash_object(first) != hash_object(second)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param(1, 1.0, id="int-float"),
        pytest.param(1, True, id="int-bool"),
        pytest.param(0.0, -0.0, id="zero-negative-zero"),
        pytest.param("1", b"1", id="str-bytes"),
        pytest.param(b"1", bytearray(b"1"), id="bytes-bytearray"),
        pytest.param([1, 2], (1, 2), id="list-tuple"),
        pytest.param([1, 2], deque([1, 2]), id="list-deque"),
        pytest.param([1, 2], {1, 2}, id="list-set"),
        pytest.param({1, 2}, frozenset({1, 2}), id="set-frozenset"),
        pytest.param({"a": 1}, OrderedDict(a=1), id="dict-ordereddict"),
        pytest.param([[1], 2], [1, [2]], id="nesting"),
        pytest.param(["ab", "c"], ["a", "bc"], id="item-boundaries"),
        pytest.param([1, 2], [2, 1], id="list-order"),
        pytest.param(OrderedDict(a=1, b=2), OrderedDict(b=2, a=1), id="ordereddict-order"),
        pytest.param(date(2024, 1, 1), datetime(2024, 1, 1), id="date-datetime"),
        pytest.param(
            datetime(2024, 1, 1), datetime(2024, 1, 1, tzinfo=timezone.utc), id="naive-aware"
        ),
        pytest.param(Path("a"), "a", id="path-str"),
        pytest.param(Color.RED, 1, id="enum-int"),
        pytest.param(Number.ONE, 1, id="intenum-int"),
        pytest.param(Label.A, "a", id="strenum-str"),
        pytest.param(Point(1, 2), (1, 2), id="namedtuple-tuple"),
        pytest.param(Point(1, 2), Other(1, 2), id="namedtuple-namedtuple"),
    ],
)
def test_hash_object_type_sensitive(first: object, second: object) -> None:
    assert hash_object(first) != hash_object(second)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param({"a": 1, "b": 2}, {"b": 2, "a": 1}, id="dict-order"),
        pytest.param({1, 2, 3}, {3, 2, 1}, id="set-order"),
        pytest.param({1: "a", "x": 2}, {"x": 2, 1: "a"}, id="mixed-key-types"),
        pytest.param(float("nan"), float("nan"), id="nan"),
        pytest.param(Point(1, 2), Point(1, 2), id="namedtuple"),
    ],
)
def test_hash_object_equal_values_hash_the_same(first: object, second: object) -> None:
    assert hash_object(first) == hash_object(second)


def test_hash_object_empty_containers_do_not_collide() -> None:
    values = [None, "", b"", [], (), {}, set(), frozenset(), deque(), bytearray()]
    assert len({hash_object(value) for value in values}) == len(values)


def test_hash_object_list_and_subclass_hash_the_same() -> None:
    class MyList(list):
        pass

    assert hash_object(MyList([1, 2])) == hash_object([1, 2])


def test_hash_object_self_referencing_list_raises() -> None:
    data: list = []
    data.append(data)
    with pytest.raises(ValueError, match="self-referencing"):
        hash_object(data)


def test_hash_object_self_referencing_dict_raises() -> None:
    data: dict = {}
    data["self"] = data
    with pytest.raises(ValueError, match="self-referencing"):
        hash_object(data)


def test_hash_object_repeated_shared_object_is_not_a_cycle() -> None:
    shared = [1, 2]
    assert isinstance(hash_object([shared, shared]), str)


def test_hash_object_recovers_after_self_reference_error() -> None:
    data: list = []
    data.append(data)
    with pytest.raises(ValueError, match="self-referencing"):
        hash_object(data)
    assert isinstance(hash_object([1, 2]), str)


def test_hash_object_lone_surrogate_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not valid UTF-8"):
        hash_object("\ud800")


def test_hash_object_ignore_unhashable_warns() -> None:
    with pytest.warns(UserWarning, match="placeholder"):
        hash_object(object(), ignore_unhashable=True)


@numpy_available
def test_hash_object_numpy_scalar_hashes_like_builtin() -> None:
    assert hash_object(np.float64(1.5)) == hash_object(1.5)
