from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from typing import NamedTuple

from coola.recursive.rebuild import rebuild_mapping, rebuild_sequence


class Point(NamedTuple):
    x: int
    y: int


class MyDict(dict):
    pass


class MyList(list):
    pass


def test_rebuild_mapping_dict() -> None:
    result = rebuild_mapping({"a": 1}, {"b": 2})
    assert type(result) is dict
    assert result == {"b": 2}


def test_rebuild_mapping_ordered_dict() -> None:
    result = rebuild_mapping(OrderedDict(a=1), {"b": 2})
    assert type(result) is OrderedDict
    assert result == {"b": 2}


def test_rebuild_mapping_subclass() -> None:
    assert type(rebuild_mapping(MyDict(a=1), {"b": 2})) is MyDict


def test_rebuild_mapping_defaultdict() -> None:
    result = rebuild_mapping(defaultdict(list, {"a": [1]}), {"b": [2]})
    assert type(result) is defaultdict
    assert result.default_factory is list
    assert result == {"b": [2]}
    assert result["missing"] == []


def test_rebuild_mapping_defaultdict_without_factory() -> None:
    result = rebuild_mapping(defaultdict(None, {"a": 1}), {"b": 2})
    assert result.default_factory is None
    assert result == {"b": 2}


def test_rebuild_sequence_list() -> None:
    result = rebuild_sequence([1, 2], [3])
    assert type(result) is list
    assert result == [3]


def test_rebuild_sequence_tuple() -> None:
    result = rebuild_sequence((1, 2), [3])
    assert type(result) is tuple
    assert result == (3,)


def test_rebuild_sequence_subclass() -> None:
    assert type(rebuild_sequence(MyList([1]), [2])) is MyList


def test_rebuild_sequence_namedtuple() -> None:
    result = rebuild_sequence(Point(1, 2), [3, 4])
    assert result == Point(3, 4)
    assert type(result) is Point


def test_rebuild_sequence_deque() -> None:
    result = rebuild_sequence(deque([1, 2]), [3, 4])
    assert type(result) is deque
    assert result == deque([3, 4])
    assert result.maxlen is None


def test_rebuild_sequence_deque_keeps_maxlen() -> None:
    result = rebuild_sequence(deque([1, 2], maxlen=3), [3, 4])
    assert result.maxlen == 3
