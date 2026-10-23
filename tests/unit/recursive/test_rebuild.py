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


def test_rebuild_mapping_dict_type() -> None:
    result = rebuild_mapping({"a": 1}, {"b": 2})
    assert type(result) is dict


def test_rebuild_mapping_dict_values() -> None:
    result = rebuild_mapping({"a": 1}, {"b": 2})
    assert result == {"b": 2}


def test_rebuild_mapping_ordered_dict_type() -> None:
    result = rebuild_mapping(OrderedDict(a=1), {"b": 2})
    assert type(result) is OrderedDict


def test_rebuild_mapping_ordered_dict_values() -> None:
    result = rebuild_mapping(OrderedDict(a=1), {"b": 2})
    assert result == {"b": 2}


def test_rebuild_mapping_subclass() -> None:
    assert type(rebuild_mapping(MyDict(a=1), {"b": 2})) is MyDict


def test_rebuild_mapping_defaultdict_type() -> None:
    result = rebuild_mapping(defaultdict(list, {"a": [1]}), {"b": [2]})
    assert type(result) is defaultdict


def test_rebuild_mapping_defaultdict_default_factory() -> None:
    result = rebuild_mapping(defaultdict(list, {"a": [1]}), {"b": [2]})
    assert result.default_factory is list


def test_rebuild_mapping_defaultdict_values() -> None:
    result = rebuild_mapping(defaultdict(list, {"a": [1]}), {"b": [2]})
    assert result == {"b": [2]}


def test_rebuild_mapping_defaultdict_default_factory_is_used() -> None:
    result = rebuild_mapping(defaultdict(list, {"a": [1]}), {"b": [2]})
    assert result["missing"] == []


def test_rebuild_mapping_defaultdict_without_factory_no_default_factory() -> None:
    result = rebuild_mapping(defaultdict(None, {"a": 1}), {"b": 2})
    assert result.default_factory is None


def test_rebuild_mapping_defaultdict_without_factory_values() -> None:
    result = rebuild_mapping(defaultdict(None, {"a": 1}), {"b": 2})
    assert result == {"b": 2}


def test_rebuild_sequence_list_type() -> None:
    result = rebuild_sequence([1, 2], [3])
    assert type(result) is list


def test_rebuild_sequence_list_values() -> None:
    result = rebuild_sequence([1, 2], [3])
    assert result == [3]


def test_rebuild_sequence_tuple_type() -> None:
    result = rebuild_sequence((1, 2), [3])
    assert type(result) is tuple


def test_rebuild_sequence_tuple_values() -> None:
    result = rebuild_sequence((1, 2), [3])
    assert result == (3,)


def test_rebuild_sequence_subclass() -> None:
    assert type(rebuild_sequence(MyList([1]), [2])) is MyList


def test_rebuild_sequence_namedtuple_values() -> None:
    result = rebuild_sequence(Point(1, 2), [3, 4])
    assert result == Point(3, 4)


def test_rebuild_sequence_namedtuple_type() -> None:
    result = rebuild_sequence(Point(1, 2), [3, 4])
    assert type(result) is Point


def test_rebuild_sequence_deque_type() -> None:
    result = rebuild_sequence(deque([1, 2]), [3, 4])
    assert type(result) is deque


def test_rebuild_sequence_deque_values() -> None:
    result = rebuild_sequence(deque([1, 2]), [3, 4])
    assert result == deque([3, 4])


def test_rebuild_sequence_deque_maxlen() -> None:
    result = rebuild_sequence(deque([1, 2]), [3, 4])
    assert result.maxlen is None


def test_rebuild_sequence_deque_keeps_maxlen() -> None:
    result = rebuild_sequence(deque([1, 2], maxlen=3), [3, 4])
    assert result.maxlen == 3
