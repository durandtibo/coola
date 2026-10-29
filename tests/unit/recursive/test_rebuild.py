from __future__ import annotations

from collections import ChainMap, OrderedDict, defaultdict, deque
from typing import Any, NamedTuple

import pytest

from coola.recursive.rebuild import rebuild_mapping, rebuild_sequence, rebuild_set


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


class MyDeque(deque):
    pass


class WithExtra(list):
    def __init__(self, items: Any, extra: object) -> None:
        super().__init__(items)
        self.extra = extra


class MySet(set):
    pass


class WithExtraDict(dict):
    def __init__(self, items: Any, extra: object) -> None:
        super().__init__(items)
        self.extra = extra


def test_rebuild_mapping_accepts_pairs() -> None:
    assert rebuild_mapping({"a": 1}, [("b", 2)]) == {"b": 2}


def test_rebuild_mapping_error_names_the_type() -> None:
    with pytest.raises(TypeError, match=r"Cannot rebuild an object of type .*WithExtraDict"):
        rebuild_mapping(WithExtraDict({"a": 1}, 0), {"b": 2})


def test_rebuild_mapping_error_tells_how_to_register_a_transformer() -> None:
    with pytest.raises(TypeError, match="register a transformer for"):
        rebuild_mapping(WithExtraDict({"a": 1}, 0), {"b": 2})


def test_rebuild_mapping_error_cause() -> None:
    with pytest.raises(TypeError) as exc_info:
        rebuild_mapping(WithExtraDict({"a": 1}, 0), {"b": 2})
    assert "missing 1 required positional argument" in str(exc_info.value.__cause__)


def test_rebuild_sequence_accepts_generator() -> None:
    assert rebuild_sequence([1], (x for x in [2, 3])) == [2, 3]


def test_rebuild_sequence_deque_subclass() -> None:
    assert type(rebuild_sequence(MyDeque([1], maxlen=2), [2, 3])) is MyDeque


def test_rebuild_sequence_error_names_the_type() -> None:
    with pytest.raises(TypeError, match=r"Cannot rebuild an object of type .*WithExtra"):
        rebuild_sequence(WithExtra([1], 0), [2])


def test_rebuild_sequence_error_cause() -> None:
    with pytest.raises(TypeError) as exc_info:
        rebuild_sequence(WithExtra([1], 0), [2])
    assert "missing 1 required positional argument" in str(exc_info.value.__cause__)


def test_rebuild_set_set_type() -> None:
    assert type(rebuild_set({1}, [2])) is set


def test_rebuild_set_frozenset_type() -> None:
    assert type(rebuild_set(frozenset({1}), [2])) is frozenset


def test_rebuild_set_subclass() -> None:
    assert type(rebuild_set(MySet({1}), [2])) is MySet


def test_rebuild_set_values() -> None:
    assert rebuild_set({1}, [2, 3]) == {2, 3}


def test_rebuild_set_duplicates_are_merged() -> None:
    assert rebuild_set({1, 2, 3}, [0, 0, 0]) == {0}


def test_rebuild_set_empty() -> None:
    assert rebuild_set({1}, []) == set()


def test_rebuild_set_unhashable_item_raises() -> None:
    with pytest.raises(TypeError, match="unhashable type"):
        rebuild_set({1}, [[1]])


def test_rebuild_set_unhashable_item_error_names_the_type() -> None:
    with pytest.raises(TypeError, match=r"Cannot rebuild an object of type frozenset from"):
        rebuild_set(frozenset({1}), [[1]])


def test_rebuild_set_unhashable_item_error_suggests_registering_the_type() -> None:
    with pytest.raises(TypeError, match=r"register a transformer for frozenset with"):
        rebuild_set(frozenset({1}), [[1]])


def test_rebuild_chain_map_is_flattened() -> None:
    result = rebuild_mapping(ChainMap({"a": 1}, {"b": 2}), {"a": 1, "b": 2})
    assert result.maps == [{"a": 1, "b": 2}]
