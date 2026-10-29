from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NamedTuple

import pytest

from coola.nested import (
    DuplicateKeyError,
    add_prefix_suffix_to_keys,
    remove_keys_containing,
    remove_keys_if,
    remove_keys_starting_with,
)
from coola.nested.keys import _KeyTransformer
from coola.recursive import get_default_registry
from tests.helpers import DEEP_NESTING_DEPTH


@dataclass
class Node:
    name: str
    children: list


class Point(NamedTuple):
    x: int
    y: dict


def is_a(key: Any) -> bool:
    return key == "a"


def nest_dicts(leaf: Any, depth: int) -> dict:
    data = leaf
    for _ in range(depth):
        data = {"a": 0, "k": data}
    return data


def make_cycle() -> dict:
    data: dict = {"a": 1}
    data["self"] = data
    return data


###############################################
#     Tests for add_prefix_suffix_to_keys     #
###############################################


def test_add_prefix_suffix_to_keys_empty_mapping() -> None:
    assert add_prefix_suffix_to_keys({}, prefix="prefix_") == {}


def test_add_prefix_suffix_to_keys_prefix_only() -> None:
    assert add_prefix_suffix_to_keys({"key1": 1, "key2": 2}, prefix="prefix_") == {
        "prefix_key1": 1,
        "prefix_key2": 2,
    }


def test_add_prefix_suffix_to_keys_suffix_only() -> None:
    assert add_prefix_suffix_to_keys({"key1": 1, "key2": 2}, suffix="_suffix") == {
        "key1_suffix": 1,
        "key2_suffix": 2,
    }


def test_add_prefix_suffix_to_keys_prefix_and_suffix() -> None:
    assert add_prefix_suffix_to_keys({"key1": 1, "key2": 2}, prefix="pre_", suffix="_suf") == {
        "pre_key1_suf": 1,
        "pre_key2_suf": 2,
    }


def test_add_prefix_suffix_to_keys_no_prefix_or_suffix() -> None:
    assert add_prefix_suffix_to_keys({"key1": 1, "key2": 2}) == {"key1": 1, "key2": 2}


def test_add_prefix_suffix_to_keys_non_string_key() -> None:
    assert add_prefix_suffix_to_keys({1: "a", "key": "b"}, prefix="prefix_") == {
        1: "a",
        "prefix_key": "b",
    }


def test_add_prefix_suffix_to_keys_not_recursive_by_default() -> None:
    assert add_prefix_suffix_to_keys({"key1": 1, "key2": {"key3": 3}}, prefix="prefix_") == {
        "prefix_key1": 1,
        "prefix_key2": {"key3": 3},
    }


def test_add_prefix_suffix_to_keys_recursive() -> None:
    assert add_prefix_suffix_to_keys(
        {"key1": 1, "key2": {"key3": 3, "key4": {"key5": 5}}},
        prefix="prefix_",
        recursive=True,
    ) == {
        "prefix_key1": 1,
        "prefix_key2": {"prefix_key3": 3, "prefix_key4": {"prefix_key5": 5}},
    }


####################################
#     Tests for remove_keys_if     #
####################################


def test_remove_keys_if_empty() -> None:
    assert remove_keys_if({}, lambda key: key == "key") == {}


def test_remove_keys_if() -> None:
    assert remove_keys_if(
        {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6},
        lambda key: isinstance(key, str) and "key" in key,
    ) == {
        "abc": 3,
        1: 5,
        (2, 3): 6,
    }


def test_remove_keys_if_non_string_predicate() -> None:
    assert remove_keys_if(
        {"a": 1, "b": 2, 1: 3, 2: 4}, lambda key: isinstance(key, int) and key % 2 == 0
    ) == {"a": 1, "b": 2, 1: 3}


def test_remove_keys_if_nested_mapping() -> None:
    assert remove_keys_if(
        {"abc": {"key": 1, "abc": 2}, "key": {"abc": 3}},
        lambda key: isinstance(key, str) and "key" in key,
    ) == {"abc": {"abc": 2}}


def test_remove_keys_if_nested_list() -> None:
    assert remove_keys_if(
        {"list": [{"key": 1, "abc": 2}, {"key": 3}], "abc": 4},
        lambda key: isinstance(key, str) and "key" in key,
    ) == {"list": [{"abc": 2}, {}], "abc": 4}


def test_remove_keys_if_nested_tuple() -> None:
    assert remove_keys_if(
        {"tuple": ({"key": 1, "abc": 2},)}, lambda key: isinstance(key, str) and "key" in key
    ) == {"tuple": ({"abc": 2},)}


def test_remove_keys_if_nested_set() -> None:
    assert remove_keys_if(
        {"set": {1, 2, 3}, "key": 4}, lambda key: isinstance(key, str) and "key" in key
    ) == {"set": {1, 2, 3}}


def test_remove_keys_if_preserves_mapping_type_values() -> None:
    out = remove_keys_if(
        OrderedDict(key=1, abc=2), lambda key: isinstance(key, str) and "key" in key
    )
    assert out == {"abc": 2}


def test_remove_keys_if_preserves_mapping_type_type() -> None:
    out = remove_keys_if(
        OrderedDict(key=1, abc=2), lambda key: isinstance(key, str) and "key" in key
    )
    assert isinstance(out, OrderedDict)


def test_remove_keys_if_preserves_namedtuple_values() -> None:
    class Point(NamedTuple):
        x: int
        y: dict

    out = remove_keys_if(
        {"point": Point(1, {"key": 2, "abc": 3})},
        lambda key: isinstance(key, str) and "key" in key,
    )
    assert out == {"point": Point(1, {"abc": 3})}


def test_remove_keys_if_preserves_namedtuple_type() -> None:
    class Point(NamedTuple):
        x: int
        y: dict

    out = remove_keys_if(
        {"point": Point(1, {"key": 2, "abc": 3})},
        lambda key: isinstance(key, str) and "key" in key,
    )
    assert isinstance(out["point"], Point)


############################################
#     Tests for remove_keys_containing     #
############################################


def test_remove_keys_containing_empty() -> None:
    assert remove_keys_containing({}, "key") == {}


def test_remove_keys_containing() -> None:
    assert remove_keys_containing(
        {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6}, "key"
    ) == {
        "abc": 3,
        1: 5,
        (2, 3): 6,
    }


def test_remove_keys_containing_another_substring() -> None:
    assert remove_keys_containing(
        {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6}, "xyz"
    ) == {
        "key": 1,
        "key.abc": 2,
        "abc": 3,
        "abc.key": 4,
        1: 5,
        (2, 3): 6,
    }


def test_remove_keys_containing_nested_mapping() -> None:
    assert remove_keys_containing({"abc": {"key": 1, "abc": 2}, "key": {"abc": 3}}, "key") == {
        "abc": {"abc": 2}
    }


def test_remove_keys_containing_nested_list() -> None:
    assert remove_keys_containing(
        {"list": [{"key": 1, "abc": 2}, {"key": 3}], "abc": 4}, "key"
    ) == {"list": [{"abc": 2}, {}], "abc": 4}


def test_remove_keys_containing_nested_tuple() -> None:
    assert remove_keys_containing({"tuple": ({"key": 1, "abc": 2},)}, "key") == {
        "tuple": ({"abc": 2},)
    }


###############################################
#     Tests for remove_keys_starting_with     #
###############################################


def test_remove_keys_starting_with_empty() -> None:
    assert remove_keys_starting_with({}, "key") == {}


def test_remove_keys_starting_with() -> None:
    assert remove_keys_starting_with(
        {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6}, "key"
    ) == {
        "abc": 3,
        "abc.key": 4,
        1: 5,
        (2, 3): 6,
    }


def test_remove_keys_starting_with_another_key() -> None:
    assert remove_keys_starting_with(
        {"key": 1, "key.abc": 2, "abc": 3, "abc.key": 4, 1: 5, (2, 3): 6}, "key."
    ) == {
        "key": 1,
        "abc": 3,
        "abc.key": 4,
        1: 5,
        (2, 3): 6,
    }


def test_remove_keys_starting_with_nested_mapping() -> None:
    assert remove_keys_starting_with({"abc": {"key": 1, "abc": 2}, "key": {"abc": 3}}, "key") == {
        "abc": {"abc": 2}
    }


def test_remove_keys_if_uses_the_predicate_of_each_call_first_predicate() -> None:
    data = {"a": 1, "b": {"a": 2, "b": 3}}
    assert remove_keys_if(data, lambda key: key == "a") == {"b": {"b": 3}}


def test_remove_keys_if_uses_the_predicate_of_each_call_second_predicate() -> None:
    data = {"a": 1, "b": {"a": 2, "b": 3}}
    assert remove_keys_if(data, lambda key: key == "b") == {"a": 1}


def test_remove_keys_if_uses_the_predicate_of_each_call_predicate_that_keeps_all() -> None:
    data = {"a": 1, "b": {"a": 2, "b": 3}}
    assert remove_keys_if(data, lambda key: False) == data  # noqa: ARG005


def test_remove_keys_if_values_equal_to_the_removed_key_are_kept() -> None:
    assert remove_keys_if({"x": "a", "a": 1, "y": ["a"]}, lambda key: key == "a") == {
        "x": "a",
        "y": ["a"],
    }


def test_add_prefix_suffix_to_keys_recursive_list_of_dicts() -> None:
    assert add_prefix_suffix_to_keys(
        {"a": [{"b": 1}, [{"c": 2}], 3]}, prefix="p_", recursive=True
    ) == {"p_a": [{"p_b": 1}, [{"p_c": 2}], 3]}


def test_add_prefix_suffix_to_keys_recursive_tuple_of_dicts() -> None:
    assert add_prefix_suffix_to_keys({"a": ({"b": 1},)}, suffix="_s", recursive=True) == {
        "a_s": ({"b_s": 1},)
    }


def test_add_prefix_suffix_to_keys_recursive_keeps_strings() -> None:
    assert add_prefix_suffix_to_keys({"a": ["abc", b"x"]}, prefix="p_", recursive=True) == {
        "p_a": ["abc", b"x"]
    }


def test_add_prefix_suffix_to_keys_not_recursive_keeps_lists() -> None:
    assert add_prefix_suffix_to_keys({"a": [{"b": 1}]}, prefix="p_") == {"p_a": [{"b": 1}]}


#############################################
#     Tests for the types and the depth     #
#############################################


def test_remove_keys_if_whole_output_of_a_mixed_structure() -> None:
    data = {
        "a": 1,
        "b": {"a": 2, "c": [{"a": 3, "d": 4}, ({"a": 5},)]},
        "e": Node("n", [{"a": 6, "f": 7}]),
        "g": {1, 2},
    }
    assert remove_keys_if(data, is_a) == {
        "b": {"c": [{"d": 4}, ({},)]},
        "e": Node("n", [{"f": 7}]),
        "g": {1, 2},
    }


def test_remove_keys_if_enters_the_dataclass() -> None:
    assert remove_keys_if({"x": Node("n", [{"a": 1, "k": 2}])}, is_a) == {
        "x": Node("n", [{"k": 2}])
    }


def test_remove_keys_if_does_not_modify_the_dataclass() -> None:
    node = Node("n", [{"a": 1}])
    remove_keys_if({"x": node}, is_a)
    assert node == Node("n", [{"a": 1}])


@pytest.mark.parametrize("mapping_type", [OrderedDict, defaultdict, MappingProxyType])
def test_remove_keys_if_keeps_the_top_level_type(mapping_type: type) -> None:
    data = (
        mapping_type({"a": 1, "b": 2})
        if mapping_type is not defaultdict
        else defaultdict(int, {"a": 1, "b": 2})
    )
    assert type(remove_keys_if(data, is_a)) is mapping_type


def test_remove_keys_if_keeps_the_type_at_depth() -> None:
    result = remove_keys_if({"p": OrderedDict(a=1, b=2)}, is_a)
    assert type(result["p"]) is OrderedDict


def test_remove_keys_if_keeps_the_default_factory() -> None:
    result = remove_keys_if({"p": defaultdict(list, {"a": 1, "b": 2})}, is_a)
    assert result["p"].default_factory is list


def test_remove_keys_if_keeps_the_mapping_proxy_at_depth() -> None:
    result = remove_keys_if({"p": MappingProxyType({"a": 1, "b": 2})}, is_a)
    assert type(result["p"]) is MappingProxyType


def test_remove_keys_if_keeps_the_named_tuple_at_depth() -> None:
    result = remove_keys_if({"p": Point(1, {"a": 1, "b": 2})}, is_a)
    assert result == {"p": Point(1, {"b": 2})}


def test_remove_keys_if_enters_the_deque() -> None:
    assert remove_keys_if({"d": deque([{"a": 1, "b": 2}])}, is_a) == {"d": deque([{"b": 2}])}


def test_remove_keys_if_keeps_the_order() -> None:
    assert list(remove_keys_if({"c": 1, "a": 2, "b": 3, "d": 4}, is_a)) == ["c", "b", "d"]


def test_remove_keys_if_does_not_modify_the_input() -> None:
    data = {"a": 1, "b": {"a": 2, "c": [{"a": 3}]}}
    remove_keys_if(data, is_a)
    assert data == {"a": 1, "b": {"a": 2, "c": [{"a": 3}]}}


def test_remove_keys_if_does_not_share_containers_with_the_input() -> None:
    data = {"b": {"c": 1}}
    assert remove_keys_if(data, is_a)["b"] is not data["b"]


def test_remove_keys_if_predicate_that_never_matches_is_the_identity() -> None:
    data = {"b": {"c": [1, (2, {"d": 3})]}, "e": OrderedDict(f=1)}
    assert remove_keys_if(data, lambda key: False) == data  # noqa: ARG005


@pytest.mark.parametrize("data", [[1], 42, "abc", None, [{"a": 1}]])
def test_remove_keys_if_non_mapping_raises(data: Any) -> None:
    with pytest.raises(TypeError, match="mapping"):
        remove_keys_if(data, is_a)


def test_remove_keys_containing_non_mapping_raises() -> None:
    with pytest.raises(TypeError, match="mapping"):
        remove_keys_containing([1], "a")


def test_remove_keys_starting_with_non_mapping_raises() -> None:
    with pytest.raises(TypeError, match="mapping"):
        remove_keys_starting_with([1], "a")


def test_remove_keys_if_cycle_raises() -> None:
    with pytest.raises(ValueError, match="Cycle detected"):
        remove_keys_if(make_cycle(), is_a)


def test_remove_keys_if_deep_structure_does_not_raise() -> None:
    result = remove_keys_if(nest_dicts({"a": 1, "b": 2}, DEEP_NESTING_DEPTH), is_a)
    for _ in range(DEEP_NESTING_DEPTH):
        result = result["k"]
    assert result == {"b": 2}


def test_remove_keys_if_predicate_error_propagates() -> None:
    def failing(key: Any) -> bool:
        del key
        msg = "failing predicate"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="failing predicate"):
        remove_keys_if({"a": 1}, failing)


def test_remove_keys_if_does_not_modify_the_default_registry() -> None:
    remove_keys_if({"a": 1}, is_a)
    assert not isinstance(get_default_registry().find_transformer(dict), _KeyTransformer)


#################################################
#     Tests for add_prefix_suffix_to_keys 2     #
#################################################


def test_add_prefix_suffix_to_keys_whole_output_recursive() -> None:
    data = {"a": 1, 2: {"b": [{"c": 3}, ({"d": 4},)]}, "e": Node("n", [{"f": 5}])}
    assert add_prefix_suffix_to_keys(data, "x_", "_y", recursive=True) == {
        "x_a_y": 1,
        2: {"x_b_y": [{"x_c_y": 3}, ({"x_d_y": 4},)]},
        "x_e_y": Node("n", [{"x_f_y": 5}]),
    }


def test_add_prefix_suffix_to_keys_recursive_keeps_the_type_at_depth() -> None:
    result = add_prefix_suffix_to_keys({"p": OrderedDict(a=1)}, "x_", recursive=True)
    assert type(result["x_p"]) is OrderedDict


def test_add_prefix_suffix_to_keys_keeps_the_top_level_type() -> None:
    assert type(add_prefix_suffix_to_keys(OrderedDict(a=1), "x_")) is OrderedDict


def test_add_prefix_suffix_to_keys_not_recursive_keeps_the_top_level_type_of_a_defaultdict() -> (
    None
):
    result = add_prefix_suffix_to_keys(defaultdict(list, {"a": 1}), "x_")
    assert result.default_factory is list


def test_add_prefix_suffix_to_keys_recursive_enters_the_dataclass() -> None:
    result = add_prefix_suffix_to_keys({"n": Node("n", [{"a": 1}])}, "x_", recursive=True)
    assert result == {"x_n": Node("n", [{"x_a": 1}])}


def test_add_prefix_suffix_to_keys_recursive_non_mapping_raises() -> None:
    with pytest.raises(TypeError, match="mapping"):
        add_prefix_suffix_to_keys([1], "x_", recursive=True)


def test_add_prefix_suffix_to_keys_non_mapping_raises() -> None:
    with pytest.raises(TypeError, match="mapping"):
        add_prefix_suffix_to_keys([1], "x_")


def test_add_prefix_suffix_to_keys_recursive_cycle_raises() -> None:
    with pytest.raises(ValueError, match="Cycle detected"):
        add_prefix_suffix_to_keys(make_cycle(), "x_", recursive=True)


def test_add_prefix_suffix_to_keys_recursive_deep_structure_does_not_raise() -> None:
    result = add_prefix_suffix_to_keys(
        nest_dicts({"a": 1}, DEEP_NESTING_DEPTH), "x_", recursive=True
    )
    for _ in range(DEEP_NESTING_DEPTH):
        result = result["x_k"]
    assert result == {"x_a": 1}


def test_add_prefix_suffix_to_keys_does_not_modify_the_input() -> None:
    data = {"a": {"b": 1}}
    add_prefix_suffix_to_keys(data, "x_", recursive=True)
    assert data == {"a": {"b": 1}}


def test_add_prefix_suffix_to_keys_options_are_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional"):
        add_prefix_suffix_to_keys({"a": 1}, "x_", "_y", True)


####################################
#     Tests for _KeyTransformer    #
####################################


def test_key_transformer_expand_rename_collision_raises() -> None:
    with pytest.raises(DuplicateKeyError, match="duplicate keys"):
        _KeyTransformer(rename=lambda key: "same").expand({"a": 1, "b": 2})  # noqa: ARG005


def test_key_transformer_expand_drop_and_rename() -> None:
    children, rebuild = _KeyTransformer(rename=str.upper, drop=is_a).expand({"a": 1, "b": 2})
    assert rebuild(children) == {"B": 2}


def test_key_transformer_expand_drop_before_rename_avoids_a_collision() -> None:
    transformer = _KeyTransformer(rename=lambda key: "k", drop=is_a)  # noqa: ARG005
    children, rebuild = transformer.expand({"a": 1, "b": 2})
    assert rebuild(children) == {"k": 2}


def test_key_transformer_repr() -> None:
    assert repr(_KeyTransformer()) == "_KeyTransformer(rename=None, drop=None)"
