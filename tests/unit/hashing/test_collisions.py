from __future__ import annotations

import contextlib
import os
from collections import OrderedDict, deque
from datetime import UTC, date, datetime, timedelta, timezone
from enum import Enum, IntEnum, StrEnum
from pathlib import Path
from typing import NamedTuple

import pytest

from coola.equality import objects_are_equal
from coola.hashing import hash_object
from coola.testing.marks import numpy_available
from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np


class Point(NamedTuple):
    x: int
    y: int


class Other(NamedTuple):
    x: int
    y: int


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
        pytest.param(date(2024, 1, 1), datetime(2024, 1, 1, tzinfo=UTC), id="date-datetime"),
        pytest.param(
            datetime(2024, 1, 1),  # noqa: DTZ001
            datetime(2024, 1, 1, tzinfo=UTC),
            id="naive-aware",
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
        pytest.param(0.0, -0.0, id="zero-negative-zero"),
        pytest.param(complex(-0.0, -0.0), 0j, id="complex-negative-zero"),
        pytest.param(
            datetime(2024, 1, 1, 12, tzinfo=UTC),
            datetime(2024, 1, 1, 13, tzinfo=timezone(timedelta(hours=1))),
            id="same-instant-different-zone",
        ),
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


def test_hash_object_recovers_after_self_reference_error_raises() -> None:
    data: list = []
    data.append(data)
    with pytest.raises(ValueError, match="self-referencing"):
        hash_object(data)


def test_hash_object_recovers_after_self_reference_error() -> None:
    data: list = []
    data.append(data)
    with contextlib.suppress(ValueError):
        hash_object(data)
    assert isinstance(hash_object([1, 2]), str)


def test_hash_object_lone_surrogate() -> None:
    assert len(hash_object("\ud800")) == 64


def test_hash_object_path_with_lone_surrogate() -> None:
    assert len(hash_object(Path(os.fsdecode(b"a\x80")))) == 64


def test_hash_object_ignore_unhashable_warns() -> None:
    with pytest.warns(UserWarning, match="placeholder"):
        hash_object(object(), ignore_unhashable=True)


@numpy_available
def test_hash_object_numpy_scalar_hashes_like_builtin() -> None:
    assert hash_object(np.float64(1.5)) == hash_object(1.5)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param({"a": 1, "b": [1, 2]}, {"b": [1, 2], "a": 1}, id="dict"),
        pytest.param(OrderedDict(a=1, b=2), OrderedDict(a=1, b=2), id="ordereddict"),
        pytest.param({1, 2, 3}, {3, 2, 1}, id="set"),
        pytest.param(0.0, -0.0, id="negative-zero"),
        pytest.param(
            datetime(2024, 1, 1, 12, tzinfo=UTC),
            datetime(2024, 1, 1, 13, tzinfo=timezone(timedelta(hours=1))),
            id="aware-datetime",
        ),
        pytest.param(Point(1, 2), Point(1, 2), id="namedtuple"),
        pytest.param([1, (2, {3})], [1, (2, {3})], id="nested"),
    ],
)
def test_hash_object_is_consistent_with_objects_are_equal(first: object, second: object) -> None:
    # Objects that coola considers equal must have the same hash.
    assert objects_are_equal(first, second)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        pytest.param({"a": 1, "b": [1, 2]}, {"b": [1, 2], "a": 1}, id="dict"),
        pytest.param(OrderedDict(a=1, b=2), OrderedDict(a=1, b=2), id="ordereddict"),
        pytest.param({1, 2, 3}, {3, 2, 1}, id="set"),
        pytest.param(0.0, -0.0, id="negative-zero"),
        pytest.param(
            datetime(2024, 1, 1, 12, tzinfo=UTC),
            datetime(2024, 1, 1, 13, tzinfo=timezone(timedelta(hours=1))),
            id="aware-datetime",
        ),
        pytest.param(Point(1, 2), Point(1, 2), id="namedtuple"),
        pytest.param([1, (2, {3})], [1, (2, {3})], id="nested"),
    ],
)
def test_hash_object_is_consistent_with_objects_are_equal_same_hash(
    first: object, second: object
) -> None:
    # Objects that coola considers equal must have the same hash.
    assert hash_object(first) == hash_object(second)


def test_hash_object_ordereddict_is_order_sensitive_but_dict_is_not_ordereddict_is_order_sensitive() -> (
    None
):
    assert hash_object(OrderedDict(a=1, b=2)) != hash_object(OrderedDict(b=2, a=1))


def test_hash_object_ordereddict_is_order_sensitive_but_dict_is_not_dict_is_not() -> None:
    assert hash_object({"a": 1, "b": 2}) == hash_object({"b": 2, "a": 1})


def test_hash_object_ordereddict_consistent_with_objects_are_equal() -> None:
    first = OrderedDict(a=1, b=2)
    second = OrderedDict(b=2, a=1)
    assert not objects_are_equal(first, second)


def test_hash_object_ordereddict_consistent_with_objects_are_equal_different_hash() -> None:
    first = OrderedDict(a=1, b=2)
    second = OrderedDict(b=2, a=1)
    assert hash_object(first) != hash_object(second)


def test_hash_object_ordereddict_subclass_is_order_sensitive() -> None:
    class MyOrderedDict(OrderedDict):
        pass

    first = MyOrderedDict(a=1, b=2)
    second = MyOrderedDict(b=2, a=1)
    assert not objects_are_equal(first, second)


def test_hash_object_ordereddict_subclass_is_order_sensitive_different_hash() -> None:
    class MyOrderedDict(OrderedDict):
        pass

    first = MyOrderedDict(a=1, b=2)
    second = MyOrderedDict(b=2, a=1)
    assert hash_object(first) != hash_object(second)


def test_hash_object_ordereddict_subclass_is_order_sensitive_same_order_same_hash() -> None:
    class MyOrderedDict(OrderedDict):
        pass

    first = MyOrderedDict(a=1, b=2)
    MyOrderedDict(b=2, a=1)
    assert hash_object(first) == hash_object(MyOrderedDict(a=1, b=2))


def test_hash_object_ordereddict_nested_order_is_checked() -> None:
    first = {"k": [OrderedDict(a=1, b=2)]}
    second = {"k": [OrderedDict(b=2, a=1)]}
    assert not objects_are_equal(first, second)


def test_hash_object_ordereddict_nested_order_is_checked_different_hash() -> None:
    first = {"k": [OrderedDict(a=1, b=2)]}
    second = {"k": [OrderedDict(b=2, a=1)]}
    assert hash_object(first) != hash_object(second)
