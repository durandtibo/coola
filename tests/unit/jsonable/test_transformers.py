from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import pytest

from coola.jsonable import (
    ConvertTransformer,
    DataclassTransformer,
    MappingTransformer,
    SequenceTransformer,
    SetTransformer,
)
from coola.jsonable.transformers import (
    _first,
    _identity,
    _sort_key,
    _sorted_elements,
    _to_dict,
    _to_key,
)
from tests.unit.recursive.helpers import expand_not_none


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Empty:
    pass


#####################################
#     Tests for ConvertTransformer  #
#####################################


def test_convert_transformer_repr() -> None:
    assert repr(ConvertTransformer(str)) == "ConvertTransformer(func=<class 'str'>)"


def test_convert_transformer_expand_children() -> None:
    children, _ = expand_not_none(ConvertTransformer(str), PurePosixPath("a/b"))
    assert children == ["a/b"]


def test_convert_transformer_expand_rebuild() -> None:
    _, rebuild = expand_not_none(ConvertTransformer(str), PurePosixPath("a/b"))
    assert rebuild(["x"]) == "x"


def test_convert_transformer_expand_does_not_convert_the_result() -> None:
    children, _ = expand_not_none(ConvertTransformer(lambda x: (x, x)), 1)
    assert children == [(1, 1)]


def test_convert_transformer_expand_same_type_raises() -> None:
    with pytest.raises(TypeError, match=r"The converter of int returned a value of the same type"):
        ConvertTransformer(lambda x: x + 1).expand(1)


def test_convert_transformer_expand_propagates_the_error_of_the_function() -> None:
    def fail(_: Any) -> Any:
        msg = "failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match=r"failed"):
        ConvertTransformer(fail).expand(1)


######################################
#     Tests for SequenceTransformer  #
######################################


def test_sequence_transformer_repr() -> None:
    assert repr(SequenceTransformer()) == "SequenceTransformer()"


@pytest.mark.parametrize(
    "data",
    [
        pytest.param([1, 2, 3], id="list"),
        pytest.param((1, 2, 3), id="tuple"),
        pytest.param(deque([1, 2, 3]), id="deque"),
        pytest.param(range(1, 4), id="range"),
    ],
)
def test_sequence_transformer_expand_children(data: Any) -> None:
    children, _ = expand_not_none(SequenceTransformer(), data)
    assert children == [1, 2, 3]


def test_sequence_transformer_expand_empty() -> None:
    children, rebuild = expand_not_none(SequenceTransformer(), ())
    assert (children, rebuild([])) == ([], [])


def test_sequence_transformer_expand_single_element() -> None:
    children, _ = expand_not_none(SequenceTransformer(), (1,))
    assert children == [1]


def test_sequence_transformer_expand_duplicates() -> None:
    children, _ = expand_not_none(SequenceTransformer(), [1, 1])
    assert children == [1, 1]


def test_sequence_transformer_rebuild_makes_a_list() -> None:
    _, rebuild = expand_not_none(SequenceTransformer(), (1, 2))
    assert rebuild(["1", "2"]) == ["1", "2"]


def test_sequence_transformer_expand_does_not_modify_the_input() -> None:
    data = [1, [2]]
    expand_not_none(SequenceTransformer(), data)
    assert data == [1, [2]]


def test_sequence_transformer_expand_children_do_not_share_the_input_list() -> None:
    data = [1, 2]
    children, _ = expand_not_none(SequenceTransformer(), data)
    assert children is not data


##################################
#     Tests for SetTransformer   #
##################################


def test_set_transformer_repr() -> None:
    assert repr(SetTransformer()) == "SetTransformer()"


@pytest.mark.parametrize("data", [{1, 2, 3}, frozenset({1, 2, 3})])
def test_set_transformer_expand_children(data: Any) -> None:
    children, _ = expand_not_none(SetTransformer(), data)
    assert sorted(children) == [1, 2, 3]


def test_set_transformer_expand_empty() -> None:
    children, rebuild = expand_not_none(SetTransformer(), set())
    assert (children, rebuild([])) == ([], [])


def test_set_transformer_expand_single_element() -> None:
    children, rebuild = expand_not_none(SetTransformer(), {1})
    assert (children, rebuild([1])) == ([1], [1])


def test_set_transformer_rebuild_sorts() -> None:
    _, rebuild = expand_not_none(SetTransformer(), {3, 1, 2})
    assert rebuild([3, 1, 2]) == [1, 2, 3]


def test_set_transformer_rebuild_keeps_the_duplicates_of_the_conversion() -> None:
    _, rebuild = expand_not_none(SetTransformer(), {1, 2})
    assert rebuild([1, 1]) == [1, 1]


#####################################
#     Tests for MappingTransformer  #
#####################################


def test_mapping_transformer_repr() -> None:
    assert repr(MappingTransformer()) == "MappingTransformer()"


def test_mapping_transformer_expand_children_are_the_keys_then_the_values() -> None:
    children, _ = expand_not_none(MappingTransformer(), {"a": 1, "b": 2})
    assert children == ["a", "b", 1, 2]


def test_mapping_transformer_expand_empty() -> None:
    children, rebuild = expand_not_none(MappingTransformer(), {})
    assert (children, rebuild([])) == ([], {})


def test_mapping_transformer_rebuild_converts_the_keys() -> None:
    _, rebuild = expand_not_none(MappingTransformer(), {"a": 1, 2: 3})
    assert rebuild(["a", 2, 1, 3]) == {"a": 1, "2": 3}


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(OrderedDict(b=2, a=1), id="ordered_dict"),
        pytest.param(defaultdict(int, b=2, a=1), id="defaultdict"),
    ],
)
def test_mapping_transformer_rebuild_makes_a_dict(data: Any) -> None:
    children, rebuild = expand_not_none(MappingTransformer(), data)
    result = rebuild(list(children))
    assert (type(result), list(result.items())) == (dict, [("b", 2), ("a", 1)])


def test_mapping_transformer_rebuild_does_not_modify_the_input() -> None:
    data = {"a": 1}
    children, rebuild = expand_not_none(MappingTransformer(), data)
    rebuild(list(children))
    assert data == {"a": 1}


#######################################
#     Tests for DataclassTransformer  #
#######################################


def test_dataclass_transformer_repr() -> None:
    assert repr(DataclassTransformer()) == "DataclassTransformer()"


def test_dataclass_transformer_expand_children() -> None:
    children, _ = expand_not_none(DataclassTransformer(), Point(1, 2))
    assert children == [1, 2]


def test_dataclass_transformer_rebuild_makes_a_dict() -> None:
    _, rebuild = expand_not_none(DataclassTransformer(), Point(1, 2))
    assert rebuild([3, 4]) == {"x": 3, "y": 4}


def test_dataclass_transformer_expand_empty() -> None:
    children, rebuild = expand_not_none(DataclassTransformer(), Empty())
    assert (children, rebuild([])) == ([], {})


def test_dataclass_transformer_expand_does_not_modify_the_input() -> None:
    data = Point(1, 2)
    expand_not_none(DataclassTransformer(), data)
    assert data == Point(1, 2)


################################
#     Tests for the helpers    #
################################


def test_first() -> None:
    assert _first([1, 2]) == 1


def test_identity() -> None:
    items = [1, 2]
    assert _identity(items) is items


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        pytest.param("a", "a", id="str"),
        pytest.param(True, "true", id="true"),
        pytest.param(False, "false", id="false"),
        pytest.param(1, "1", id="int"),
        pytest.param(1.5, "1.5", id="float"),
        pytest.param(None, "null", id="none"),
        pytest.param(float("inf"), "Infinity", id="inf"),
    ],
)
def test_to_key(key: Any, expected: str) -> None:
    assert _to_key(key) == expected


@pytest.mark.parametrize("key", [[1, 2], {"a": 1}, b"a", object()])
def test_to_key_raises(key: Any) -> None:
    with pytest.raises(TypeError, match=r"A JSON key must be a str, bool, int, float or None"):
        _to_key(key)


def test_to_key_raises_names_the_type() -> None:
    with pytest.raises(TypeError, match=r"a key of type list"):
        _to_key([1])


def test_to_dict() -> None:
    assert _to_dict(["a", 2, 1, 3]) == {"a": 1, "2": 3}


def test_to_dict_empty() -> None:
    assert _to_dict([]) == {}


def test_to_dict_keeps_the_order() -> None:
    assert list(_to_dict(["b", "a", 1, 2])) == ["b", "a"]


def test_to_dict_duplicate_keys_raises() -> None:
    with pytest.raises(ValueError, match=r"Two keys give the same JSON key: '1'"):
        _to_dict([1, "1", "a", "b"])


def test_to_dict_invalid_key_raises() -> None:
    with pytest.raises(TypeError, match=r"a key of type list"):
        _to_dict([[1], "a"])


def test_sort_key() -> None:
    assert _sort_key(1) == ("int", "1")


def test_sorted_elements_by_value() -> None:
    assert _sorted_elements({3, 1, 2}) == [1, 2, 3]


def test_sorted_elements_empty() -> None:
    assert _sorted_elements(set()) == []


def test_sorted_elements_mixed_types() -> None:
    assert _sorted_elements({"b", 1, "a", 2}) == [1, 2, "a", "b"]


def test_sorted_elements_nan_uses_the_type_and_the_repr() -> None:
    assert repr(_sorted_elements([float("nan"), 1.0])) == "[1.0, nan]"


def test_sorted_elements_nan_does_not_depend_on_the_input_order() -> None:
    nan = float("nan")
    assert repr(_sorted_elements([nan, 2.0, 1.0])) == repr(_sorted_elements([1.0, nan, 2.0]))


def test_sorted_elements_lists() -> None:
    assert _sorted_elements([[2], [1]]) == [[1], [2]]


def test_sorted_elements_does_not_depend_on_the_iteration_order() -> None:
    assert _sorted_elements(iter(["b", "a"])) == _sorted_elements(iter(["a", "b"]))
