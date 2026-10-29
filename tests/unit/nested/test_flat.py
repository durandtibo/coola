from __future__ import annotations

import contextlib
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from unittest.mock import Mock

import pytest

from coola.equality import objects_are_equal
from coola.nested import from_flat_dict, to_flat_dict
from coola.testing.marks import numpy_available, torch_available
from coola.utils.imports import is_numpy_available, is_torch_available
from tests.helpers import DEEP_NESTING_DEPTH

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()

####################################
#     Tests for from_flat_dict     #
####################################


def test_from_flat_dict_empty() -> None:
    assert from_flat_dict({}) == {}


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param({"a": 1}, {"a": 1}, id="single-key"),
        pytest.param(
            {"bool": False, "float": 3.5, "int": 2, "str": "abc"},
            {"bool": False, "float": 3.5, "int": 2, "str": "abc"},
            id="already-flat",
        ),
        pytest.param(
            {"a": "a", "b.c": "c"},
            {"a": "a", "b": {"c": "c"}},
            id="one-level-nesting",
        ),
        pytest.param(
            {"d.e.f": "f"},
            {"d": {"e": {"f": "f"}}},
            id="two-level-nesting",
        ),
        pytest.param(
            # Two flat keys sharing a prefix must merge into the same sub-dict.
            {"module.component.float": 3.5, "module.component.int": 2},
            {"module": {"component": {"float": 3.5, "int": 2}}},
            id="sibling-keys-merged",
        ),
        pytest.param(
            {
                "str": "def",
                "module.component.float": 3.5,
                "module.component.int": 2,
            },
            {"str": "def", "module": {"component": {"float": 3.5, "int": 2}}},
            id="mixed-depth",
        ),
        pytest.param(
            {
                "module.component_a.float": 3.5,
                "module.component_a.int": 2,
                "module.component_b.param_a": 1,
                "module.component_b.param_b": 2,
                "module.str": "abc",
            },
            {
                "module": {
                    "component_a": {"float": 3.5, "int": 2},
                    "component_b": {"param_a": 1, "param_b": 2},
                    "str": "abc",
                }
            },
            id="deeply-nested",
        ),
    ],
)
def test_from_flat_dict(data: dict, expected: dict) -> None:
    assert from_flat_dict(data) == expected


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            # Keys like "0", "1" from to_flat_dict(list) are kept as strings;
            # from_flat_dict always returns a plain dict, never a list.
            {"0": 2, "1": "abc", "2": True, "3": 3.5},
            {"0": 2, "1": "abc", "2": True, "3": 3.5},
            id="top-level",
        ),
        pytest.param(
            {"module.0": 2, "module.1": "abc", "module.2": True},
            {"module": {"0": 2, "1": "abc", "2": True}},
            id="one-level-nested",
        ),
        pytest.param(
            {
                "module.0.0": 1,
                "module.0.1": 2,
                "module.0.2": 3,
                "module.1.bool": True,
                "str": "abc",
            },
            {
                "module": {"0": {"0": 1, "1": 2, "2": 3}, "1": {"bool": True}},
                "str": "abc",
            },
            id="deeply-nested",
        ),
    ],
)
def test_from_flat_dict_integer_string_keys(data: dict, expected: dict) -> None:
    assert from_flat_dict(data) == expected


@pytest.mark.parametrize("separator", [".", "/", "@", "[SEP]"])
def test_from_flat_dict_separator(separator: str) -> None:
    flat = {
        f"module{separator}component{separator}float": 3.5,
        f"module{separator}component{separator}int": 2,
        "str": "def",
    }
    assert from_flat_dict(flat, separator=separator) == {
        "str": "def",
        "module": {"component": {"float": 3.5, "int": 2}},
    }


def test_from_flat_dict_separator_absent_from_keys() -> None:
    # When the separator never appears in any key, the dict is returned as-is.
    assert from_flat_dict({"a": 1, "b": 2}, separator="/") == {"a": 1, "b": 2}


def test_from_flat_dict_empty_separator_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        from_flat_dict({"a.b": 1}, separator="")


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            # "a" is first set to a scalar, then "a.b" tries to nest under it.
            {"a": 1, "a.b": 2},
            id="scalar-then-nested",
        ),
        pytest.param(
            # "a.b" creates a nested dict at "a", then "a" tries to overwrite it.
            {"a.b": 2, "a": 1},
            id="nested-then-scalar",
        ),
    ],
)
def test_from_flat_dict_key_conflict_raises(data: dict) -> None:
    with pytest.raises(ValueError, match="conflict"):
        from_flat_dict(data)


@torch_available
def test_from_flat_dict_tensor_passthrough() -> None:
    tensor = torch.ones(2, 3)
    assert objects_are_equal(
        from_flat_dict({"model.weight": tensor}),
        {"model": {"weight": torch.ones(2, 3)}},
    )


@numpy_available
def test_from_flat_dict_numpy_array_passthrough() -> None:
    array = np.zeros((2, 3))
    assert objects_are_equal(
        from_flat_dict({"data.features": array}),
        {"data": {"features": np.zeros((2, 3))}},
    )


@pytest.mark.parametrize(
    "original",
    [
        pytest.param({"bool": False, "float": 3.5, "int": 2, "str": "abc"}, id="flat"),
        pytest.param(
            {"str": "def", "module": {"component": {"float": 3.5, "int": 2}}},
            id="nested",
        ),
        pytest.param(
            {
                "module": {
                    "component_a": {"float": 3.5, "int": 2},
                    "component_b": {"param_a": 1, "param_b": 2},
                    "str": "abc",
                }
            },
            id="deeply-nested",
        ),
    ],
)
def test_round_trip(original: dict) -> None:
    assert from_flat_dict(to_flat_dict(original)) == original


@pytest.mark.parametrize("separator", [".", "/", "@", "[SEP]"])
def test_round_trip_separator(separator: str) -> None:
    original = {"a": {"b": {"c": 1}}, "x": 2}
    assert (
        from_flat_dict(to_flat_dict(original, separator=separator), separator=separator) == original
    )


@torch_available
def test_round_trip_tensor() -> None:
    original = {"model": {"weight": torch.ones(2, 3), "bias": torch.zeros(3)}}
    assert objects_are_equal(from_flat_dict(to_flat_dict(original)), original)


@numpy_available
def test_round_trip_numpy_array() -> None:
    original = {"data": {"x": np.ones(4), "y": np.zeros(4)}}
    assert objects_are_equal(from_flat_dict(to_flat_dict(original)), original)


##################################
#     Tests for to_flat_dict     #
##################################


def test_to_flat_dict_empty_dict() -> None:
    assert to_flat_dict({}) == {}


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(
            {"bool": False, "float": 3.5, "int": 2, "str": "abc"},
            {"bool": False, "float": 3.5, "int": 2, "str": "abc"},
            id="flat-dict",
        ),
        pytest.param(
            {"a": "a", "b": {"c": "c"}, "d": {"e": {"f": "f"}}},
            {"a": "a", "b.c": "c", "d.e.f": "f"},
            id="nested-dict",
        ),
        pytest.param(
            # An empty nested dict is kept as a leaf value.
            {"module": {}, "str": "abc"},
            {"module": {}, "str": "abc"},
            id="empty-nested-dict-kept",
        ),
        pytest.param(
            {
                "module": {
                    "component_a": {"float": 3.5, "int": 2},
                    "component_b": {"param_a": 1, "param_b": 2},
                    "str": "abc",
                }
            },
            {
                "module.component_a.float": 3.5,
                "module.component_a.int": 2,
                "module.component_b.param_a": 1,
                "module.component_b.param_b": 2,
                "module.str": "abc",
            },
            id="deeply-nested-dict",
        ),
        pytest.param(
            # Integer keys should be converted to strings, not left as ints.
            {0: "zero", 1: {"nested": "val"}},
            {"0": "zero", "1.nested": "val"},
            id="non-string-dict-keys-stringified",
        ),
        pytest.param(
            {"module": [2, "abc", True, 3.5], "str": "abc"},
            {"module.0": 2, "module.1": "abc", "module.2": True, "module.3": 3.5, "str": "abc"},
            id="dict-with-list",
        ),
        pytest.param(
            {"module": (2, "abc", True, 3.5), "str": "abc"},
            {"module.0": 2, "module.1": "abc", "module.2": True, "module.3": 3.5, "str": "abc"},
            id="dict-with-tuple",
        ),
        pytest.param(
            {"module": [[1, 2, 3], {"bool": True}], "str": "abc"},
            {
                "module.0.0": 1,
                "module.0.1": 2,
                "module.0.2": 3,
                "module.1.bool": True,
                "str": "abc",
            },
            id="nested-list-and-dict",
        ),
        pytest.param(
            {"module": ([1, 2, 3], {"bool": True}), "str": "abc"},
            {
                "module.0.0": 1,
                "module.0.1": 2,
                "module.0.2": 3,
                "module.1.bool": True,
                "str": "abc",
            },
            id="nested-tuple-and-dict",
        ),
    ],
)
def test_to_flat_dict(data: object, expected: dict) -> None:
    assert to_flat_dict(data) == expected


def test_to_flat_dict_ordered_dict() -> None:
    # OrderedDict is a Mapping but not a plain dict.
    data = OrderedDict([("a", 1), ("b", OrderedDict([("c", 2)]))])
    assert to_flat_dict(data) == {"a": 1, "b.c": 2}


def test_to_flat_dict_deque() -> None:
    # deque is a Sequence but not a list or tuple.
    assert to_flat_dict({"items": deque([1, 2, 3])}) == {
        "items.0": 1,
        "items.1": 2,
        "items.2": 3,
    }


def test_to_flat_dict_string_values_not_iterated() -> None:
    # str is a Sequence; string values must be treated as leaves, not iterated.
    assert to_flat_dict({"a": "hello", "b": {"c": "world"}}) == {"a": "hello", "b.c": "world"}


def test_to_flat_dict_bare_list() -> None:
    # A list passed as the top-level value (no prefix) should be flattened.
    assert to_flat_dict([2, "abc", True, 3.5]) == {"0": 2, "1": "abc", "2": True, "3": 3.5}


@pytest.mark.parametrize(
    ("data", "prefix", "expected"),
    [
        pytest.param({"a": 1, "b": 2}, "root", {"root.a": 1, "root.b": 2}, id="flat-dict"),
        pytest.param({"a": {"b": 1}}, "root", {"root.a.b": 1}, id="nested-dict"),
        pytest.param([10, 20], "items", {"items.0": 10, "items.1": 20}, id="list"),
        pytest.param(42, "answer", {"answer": 42}, id="scalar-leaf"),
    ],
)
def test_to_flat_dict_prefix(data: object, prefix: str, expected: dict) -> None:
    assert to_flat_dict(data, prefix=prefix) == expected


@pytest.mark.parametrize("separator", [".", "/", "@", "[SEP]"])
def test_to_flat_dict_separator_dict(separator: str) -> None:
    result = to_flat_dict(
        {"str": "def", "module": {"component": {"float": 3.5, "int": 2}}},
        separator=separator,
    )
    assert result == {
        f"module{separator}component{separator}float": 3.5,
        f"module{separator}component{separator}int": 2,
        "str": "def",
    }


@pytest.mark.parametrize("separator", [".", "/"])
def test_to_flat_dict_separator_list(separator: str) -> None:
    assert to_flat_dict({"items": [1, 2]}, separator=separator) == {
        f"items{separator}0": 1,
        f"items{separator}1": 2,
    }


@pytest.mark.parametrize(
    ("data", "to_str", "expected"),
    [
        pytest.param(
            {"module": [[1, 2, 3], {"bool": True}], "str": "abc"},
            list,
            {"module": "[[1, 2, 3], {'bool': True}]", "str": "abc"},
            id="single-type-list",
        ),
        pytest.param(
            {"module": (2, "abc", True, 3.5), "str": "abc"},
            tuple,
            {"module": "(2, 'abc', True, 3.5)", "str": "abc"},
            id="single-type-tuple",
        ),
        pytest.param(
            {"module1": (2, "abc", True, 3.5), "module2": [1, 2, 3], "str": "abc"},
            (list, tuple),
            {"module1": "(2, 'abc', True, 3.5)", "module2": "[1, 2, 3]", "str": "abc"},
            id="multiple-types",
        ),
        pytest.param(
            # Explicitly passing None should recurse into all containers.
            {"a": [1, 2]},
            None,
            {"a.0": 1, "a.1": 2},
            id="none-is-noop",
        ),
    ],
)
def test_to_flat_dict_to_str(data: object, to_str: object, expected: dict) -> None:
    assert to_flat_dict(data, to_str=to_str) == expected  # ty: ignore[invalid-argument-type]


def test_to_flat_dict_to_str_with_prefix() -> None:
    # A to_str-matched value at the top level still requires a prefix.
    assert to_flat_dict([1, 2, 3], prefix="items", to_str=list) == {"items": "[1, 2, 3]"}


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(42, id="bare-scalar"),
        pytest.param([1, 2, 3], id="to-str-match-without-prefix"),
    ],
)
def test_to_flat_dict_none_key_raises(data: object) -> None:
    with pytest.raises(ValueError, match="None key"):
        to_flat_dict(data, to_str=list)


@torch_available
def test_to_flat_dict_tensor_passthrough() -> None:
    tensor = torch.ones(2, 3)
    assert objects_are_equal(to_flat_dict({"tensor": tensor}), {"tensor": torch.ones(2, 3)})


@torch_available
def test_to_flat_dict_tensor_mixed_with_scalars_scalar() -> None:
    tensor = torch.zeros(3)
    result = to_flat_dict({"a": 1, "b": tensor})
    assert result["a"] == 1


@torch_available
def test_to_flat_dict_tensor_mixed_with_scalars_tensor() -> None:
    tensor = torch.zeros(3)
    result = to_flat_dict({"a": 1, "b": tensor})
    assert objects_are_equal(result["b"], tensor)


@numpy_available
def test_to_flat_dict_numpy_array_passthrough() -> None:
    array = np.zeros((2, 3))
    assert objects_are_equal(to_flat_dict({"array": array}), {"array": np.zeros((2, 3))})


@numpy_available
def test_to_flat_dict_numpy_array_mixed_with_scalars_scalar() -> None:
    array = np.ones(4)
    result = to_flat_dict({"x": "hello", "y": array})
    assert result["x"] == "hello"


@numpy_available
def test_to_flat_dict_numpy_array_mixed_with_scalars_array() -> None:
    array = np.ones(4)
    result = to_flat_dict({"x": "hello", "y": array})
    assert objects_are_equal(result["y"], array)


def test_from_flat_dict_none_leaf_then_nested_raises() -> None:
    with pytest.raises(ValueError, match="conflict"):
        from_flat_dict({"a": None, "a.b": 1})


def test_from_flat_dict_non_string_key_raises() -> None:
    with pytest.raises(TypeError, match="must be strings"):
        from_flat_dict({1: "x"})  # ty: ignore[invalid-argument-type]


def test_to_flat_dict_empty_containers_are_kept() -> None:
    assert to_flat_dict({"a": {}, "b": [], "c": [1]}) == {"a": {}, "b": [], "c.0": 1}


def test_to_flat_dict_empty_top_level() -> None:
    assert to_flat_dict({}) == {}


def test_to_flat_dict_key_collision_raises() -> None:
    with pytest.raises(ValueError, match="collision"):
        to_flat_dict({1: "x", "1": "y"})


def test_to_flat_dict_separator_collision_raises() -> None:
    with pytest.raises(ValueError, match="collision"):
        to_flat_dict({"a.b": 1, "a": {"b": 2}})


def test_flat_dict_round_trip_with_empty_containers() -> None:
    data = {"a": {}, "b": {"c": 1, "d": {}}}
    assert from_flat_dict(to_flat_dict(data)) == data


@pytest.mark.parametrize("value", [b"xy", bytearray(b"xy"), b""])
def test_to_flat_dict_bytes_are_leaves(value: bytes) -> None:
    result = to_flat_dict({"a": value, "b": [value]})
    assert objects_are_equal(result, {"a": value, "b.0": value})


def test_to_flat_dict_bytes_with_prefix() -> None:
    assert objects_are_equal(to_flat_dict(b"xy", prefix="a"), {"a": b"xy"})


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param({"a.0": 1, "a.1": 2}, {"a": [1, 2]}, id="list"),
        pytest.param({"a.0.0": 1, "a.0.1": 2, "a.1.x": 3}, {"a": [[1, 2], {"x": 3}]}, id="nested"),
        pytest.param({"0": 1, "1": 2}, {"0": 1, "1": 2}, id="top level stays a dict"),
        pytest.param({"a.0": 1, "a.2": 2}, {"a": {"0": 1, "2": 2}}, id="gap in the indices"),
        pytest.param({"a.1": 1, "a.2": 2}, {"a": {"1": 1, "2": 2}}, id="indices not from 0"),
        pytest.param({"a.0": 1, "a.x": 2}, {"a": {"0": 1, "x": 2}}, id="mixed keys"),
        pytest.param({"a.1": "b", "a.0": "a"}, {"a": ["a", "b"]}, id="unordered keys"),
        pytest.param({"a.00": 1}, {"a": {"00": 1}}, id="not a canonical index"),
        pytest.param({"a": []}, {"a": []}, id="empty list leaf"),
        pytest.param({}, {}, id="empty"),
    ],
)
def test_from_flat_dict_restore_lists(data: dict, expected: dict) -> None:
    assert objects_are_equal(from_flat_dict(data, restore_lists=True), expected)


def test_from_flat_dict_restore_lists_is_off_by_default() -> None:
    assert objects_are_equal(from_flat_dict({"a.0": 1}), {"a": {"0": 1}})


def test_from_flat_dict_restore_lists_separator() -> None:
    assert objects_are_equal(
        from_flat_dict({"a/0": 1, "a/1": 2}, separator="/", restore_lists=True), {"a": [1, 2]}
    )


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3]},
        {"a": [[1, 2], {"b": [3]}], "c": 4},
        {"a": {"b": {"c": [1, {"d": 2}]}}},
        {"a": [], "b": {}},
    ],
)
def test_from_flat_dict_restore_lists_round_trip(data: dict) -> None:
    assert objects_are_equal(from_flat_dict(to_flat_dict(data), restore_lists=True), data)


def test_to_flat_dict_self_referencing_list_raises() -> None:
    data: list = [1]
    data.append(data)
    with pytest.raises(ValueError, match="Cycle detected"):
        to_flat_dict(data, prefix="x")


def test_to_flat_dict_self_referencing_dict_raises() -> None:
    data: dict = {}
    data["a"] = data
    with pytest.raises(ValueError, match="Cycle detected"):
        to_flat_dict(data)


def test_to_flat_dict_shared_container_is_not_a_cycle() -> None:
    shared = [1]
    assert to_flat_dict({"a": shared, "b": shared}) == {"a.0": 1, "b.0": 1}


def test_from_flat_dict_dict_leaf_then_nested_raises() -> None:
    with pytest.raises(ValueError, match="conflict"):
        from_flat_dict({"a": {"x": 1}, "a.y": 2})


def test_from_flat_dict_dict_leaf_then_nested_does_not_mutate_input_raises() -> None:
    leaf = {"x": 1}
    with pytest.raises(ValueError, match="conflict"):
        from_flat_dict({"a": leaf, "a.y": 2})


def test_from_flat_dict_dict_leaf_then_nested_does_not_mutate_input_input_is_not_mutated() -> None:
    leaf = {"x": 1}
    with contextlib.suppress(ValueError):
        from_flat_dict({"a": leaf, "a.y": 2})
    assert leaf == {"x": 1}


def test_from_flat_dict_dict_value_is_not_converted_by_restore_lists() -> None:
    assert from_flat_dict({"a": {"0": 1}}, restore_lists=True) == {"a": {"0": 1}}


#######################################
#     Tests for the escape option     #
#######################################


def test_to_flat_dict_escape_key_with_separator() -> None:
    assert to_flat_dict({"a.b": 1}, escape=True) == {"a\\.b": 1}


def test_to_flat_dict_escape_nested_and_dotted_keys_are_different() -> None:
    assert to_flat_dict({"a.b": 1}, escape=True) != to_flat_dict({"a": {"b": 1}}, escape=True)


def test_to_flat_dict_escape_nested_key_is_unchanged() -> None:
    assert to_flat_dict({"a": {"b": 1}}, escape=True) == {"a.b": 1}


def test_to_flat_dict_escape_backslash() -> None:
    assert to_flat_dict({"a\\b": 1}, escape=True) == {"a\\\\b": 1}


def test_to_flat_dict_escape_multi_character_separator() -> None:
    assert to_flat_dict({"a::b": {"c": 1}}, separator="::", escape=True) == {"a\\::b::c": 1}


def test_to_flat_dict_escape_key_in_a_list() -> None:
    assert to_flat_dict([{"a.b": 1}], escape=True) == {"0.a\\.b": 1}


def test_to_flat_dict_escape_prefix_is_not_escaped() -> None:
    assert to_flat_dict({"a": 1}, prefix="x.y", escape=True) == {"x.y.a": 1}


def test_to_flat_dict_escape_false_is_the_default() -> None:
    assert to_flat_dict({"a.b": 1}) == {"a.b": 1}


def test_from_flat_dict_escape_key_with_separator() -> None:
    assert from_flat_dict({"a\\.b": 1}, escape=True) == {"a.b": 1}


def test_from_flat_dict_escape_nested_key() -> None:
    assert from_flat_dict({"a.b": 1}, escape=True) == {"a": {"b": 1}}


def test_from_flat_dict_escape_backslash() -> None:
    assert from_flat_dict({"a\\\\b": 1}, escape=True) == {"a\\b": 1}


def test_from_flat_dict_escape_backslash_before_separator() -> None:
    assert from_flat_dict({"a\\\\.b": 1}, escape=True) == {"a\\": {"b": 1}}


def test_from_flat_dict_escape_lone_backslash_is_kept() -> None:
    assert from_flat_dict({"a\\b": 1}, escape=True) == {"a\\b": 1}


def test_from_flat_dict_escape_trailing_backslash_is_kept() -> None:
    assert from_flat_dict({"a\\": 1}, escape=True) == {"a\\": 1}


def test_from_flat_dict_escape_multi_character_separator() -> None:
    assert from_flat_dict({"a\\::b::c": 1}, separator="::", escape=True) == {"a::b": {"c": 1}}


def test_from_flat_dict_escape_false_splits_every_separator() -> None:
    assert from_flat_dict({"a\\.b": 1}) == {"a\\": {"b": 1}}


@pytest.mark.parametrize(
    "data",
    [
        {"a.b": 1, "c": {"d": 2}},
        {"a\\b": 1},
        {"a\\.b": {"c.d": [1, {"e.f": 2}]}},
        {"": {"": 1}},
        {"..": 1},
    ],
)
def test_escape_round_trip(data: dict[str, Any]) -> None:
    assert from_flat_dict(to_flat_dict(data, escape=True), escape=True, restore_lists=True) == data


def test_from_flat_dict_separator_is_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional argument"):
        from_flat_dict({"a/b": 1}, "/")  # ty: ignore[too-many-positional-arguments]


def test_from_flat_dict_restore_lists_is_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional argument"):
        from_flat_dict({"a.0": 1}, ".", True)  # ty: ignore[too-many-positional-arguments]


def test_from_flat_dict_dict_value_is_the_object_of_the_caller() -> None:
    value = {"0": 1}
    assert from_flat_dict({"a": value}, restore_lists=True)["a"] is value


def test_from_flat_dict_dict_value_is_not_converted_inside_a_node() -> None:
    assert from_flat_dict({"a.b": {"0": 1}, "a.c": 2}, restore_lists=True) == {
        "a": {"b": {"0": 1}, "c": 2}
    }


def test_from_flat_dict_list_value_is_the_object_of_the_caller() -> None:
    value = [{"0": 1}]
    assert from_flat_dict({"a": value})["a"] is value


def test_from_flat_dict_root_is_a_plain_dict() -> None:
    assert type(from_flat_dict({"0": 1, "1": 2}, restore_lists=True)) is dict


def test_from_flat_dict_nodes_are_plain_dicts() -> None:
    assert type(from_flat_dict({"a.b": 1})["a"]) is dict


def test_from_flat_dict_root_with_integer_keys_is_a_dict_with_restore_lists() -> None:
    assert from_flat_dict({"0": 1, "1": 2}, restore_lists=True) == {"0": 1, "1": 2}


def test_from_flat_dict_empty_with_restore_lists() -> None:
    assert from_flat_dict({}, restore_lists=True) == {}


def test_from_flat_dict_mapping_input() -> None:
    assert from_flat_dict(MappingProxyType({"a.b": 1})) == {"a": {"b": 1}}


def test_from_flat_dict_does_not_modify_the_input() -> None:
    data = {"a.0": 1, "a.1": {"0": 2}}
    from_flat_dict(data, restore_lists=True)
    assert data == {"a.0": 1, "a.1": {"0": 2}}


def test_from_flat_dict_deep_structure_does_not_raise() -> None:
    result = from_flat_dict({".".join(["0"] * DEEP_NESTING_DEPTH): 1}, restore_lists=True)
    for _ in range(DEEP_NESTING_DEPTH):
        result = result["0"] if isinstance(result, dict) else result[0]
    assert result == 1


def test_from_flat_dict_deep_structure_without_restore_lists_does_not_raise() -> None:
    result = from_flat_dict({".".join(["a"] * DEEP_NESTING_DEPTH): 1})
    for _ in range(DEEP_NESTING_DEPTH):
        result = result["a"]
    assert result == 1


@dataclass
class Node:
    name: str
    children: list


@dataclass
class Empty:
    pass


def test_to_flat_dict_enters_the_dataclass() -> None:
    assert to_flat_dict({"x": Node("n", [1, {"k": 2}])}) == {
        "x.name": "n",
        "x.children.0": 1,
        "x.children.1.k": 2,
    }


def test_to_flat_dict_root_dataclass() -> None:
    assert to_flat_dict(Node("n", [1])) == {"name": "n", "children.0": 1}


def test_to_flat_dict_empty_dataclass_is_kept() -> None:
    assert to_flat_dict({"x": Empty()}) == {"x": Empty()}


def test_to_flat_dict_dataclass_class_is_a_leaf() -> None:
    assert to_flat_dict({"x": Node}) == {"x": Node}


def test_to_flat_dict_dataclass_in_to_str() -> None:
    assert to_flat_dict({"x": Empty()}, to_str=Empty) == {"x": str(Empty())}


def test_to_flat_dict_dataclass_in_a_cycle_raises() -> None:
    node = Node("n", [])
    node.children.append(node)
    with pytest.raises(ValueError, match="Cycle detected"):
        to_flat_dict(node)


def test_to_flat_dict_set_is_a_leaf() -> None:
    assert to_flat_dict({"a": {1, 2}}) == {"a": {1, 2}}


def test_to_flat_dict_frozenset_is_a_leaf() -> None:
    assert to_flat_dict({"a": frozenset({1})}) == {"a": frozenset({1})}


def test_to_flat_dict_set_in_a_dataclass_is_a_leaf() -> None:
    assert to_flat_dict(Node("n", [{1}])) == {"name": "n", "children.0": {1}}


def test_to_flat_dict_shared_container_has_its_entries_twice() -> None:
    shared = [1]
    assert to_flat_dict([shared, shared]) == {"0.0": 1, "1.0": 1}


def test_to_flat_dict_empty_containers_whole_output() -> None:
    assert to_flat_dict({"a": {}, "b": [], "c": (), "d": [1, {}]}) == {
        "a": {},
        "b": [],
        "c": (),
        "d.0": 1,
        "d.1": {},
    }


def test_to_flat_dict_empty_root_is_empty() -> None:
    assert to_flat_dict({}) == {}


def test_to_flat_dict_empty_root_with_prefix() -> None:
    assert to_flat_dict([], prefix="p") == {"p": []}


def test_to_flat_dict_root_leaf_with_prefix() -> None:
    assert to_flat_dict(42, prefix="p") == {"p": 42}


def test_to_flat_dict_root_leaf_without_prefix_raises() -> None:
    with pytest.raises(ValueError, match="Provide a non-None prefix"):
        to_flat_dict(42)


def test_to_flat_dict_root_to_str_without_prefix_raises() -> None:
    with pytest.raises(ValueError, match="matched by to_str"):
        to_flat_dict([1], to_str=list)


def test_to_flat_dict_root_to_str_with_prefix() -> None:
    assert to_flat_dict([1], prefix="p", to_str=list) == {"p": "[1]"}


def test_to_flat_dict_prefix_is_used_as_it_is() -> None:
    assert to_flat_dict({"a": 1}, prefix="") == {".a": 1}


def test_to_flat_dict_range_is_a_leaf() -> None:
    assert to_flat_dict({"a": range(2)}) == {"a": range(2)}


def test_to_flat_dict_iterator_is_a_leaf() -> None:
    iterator = iter([1])
    assert to_flat_dict({"a": iterator}) == {"a": iterator}


def test_to_flat_dict_str_bytes_are_leaves() -> None:
    assert to_flat_dict({"a": "xy", "b": b"xy"}) == {"a": "xy", "b": b"xy"}


def test_to_flat_dict_mapping_proxy() -> None:
    assert to_flat_dict(MappingProxyType({"a": [1]})) == {"a.0": 1}


def test_to_flat_dict_named_tuple_and_deque() -> None:
    assert to_flat_dict({"a": deque([1, 2])}) == {"a.0": 1, "a.1": 2}


def test_to_flat_dict_to_str_abstract_type_covers_the_registered_subclasses() -> None:
    assert to_flat_dict([{"b": 1}, [1]], to_str=Mapping) == {"0": "{'b': 1}", "1.0": 1}


def test_to_flat_dict_to_str_abstract_sequence_type() -> None:
    assert to_flat_dict({"a": (1, 2), "b": [3]}, to_str=Sequence) == {"a": "(1, 2)", "b": "[3]"}


def test_to_flat_dict_to_str_subclass_of_a_registered_type() -> None:
    assert to_flat_dict([OrderedDict(b=1)], to_str=dict) == {"0": "OrderedDict({'b': 1})"}


def test_to_flat_dict_to_str_scalar_type() -> None:
    assert to_flat_dict({"a": 1, "b": "x"}, to_str=int) == {"a": "1", "b": "x"}


def test_to_flat_dict_to_str_tuple_of_types() -> None:
    assert to_flat_dict({"a": [1], "b": (2,)}, to_str=(list, tuple)) == {"a": "[1]", "b": "(2,)"}


def test_to_flat_dict_to_str_does_not_change_the_default_registry() -> None:
    to_flat_dict({"a": [1]}, to_str=list)
    assert to_flat_dict({"a": [1]}) == {"a.0": 1}


def test_to_flat_dict_non_str_keys_are_written_with_str() -> None:
    assert to_flat_dict({1: {(2, 3): "x"}}) == {"1.(2, 3)": "x"}


def test_to_flat_dict_escape_applies_to_the_dataclass_fields_and_keys() -> None:
    assert to_flat_dict({"a.b": Node("n", [])}, escape=True) == {
        "a\\.b.name": "n",
        "a\\.b.children": [],
    }


def test_to_flat_dict_options_are_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional"):
        to_flat_dict({"a": 1}, "p")  # ty: ignore[too-many-positional-arguments]


def test_to_flat_dict_deep_structure_does_not_raise() -> None:
    data: Any = 1
    for _ in range(DEEP_NESTING_DEPTH):
        data = {"a": data}
    assert to_flat_dict(data) == {".".join(["a"] * DEEP_NESTING_DEPTH): 1}


def test_to_flat_dict_deep_list_structure_does_not_raise() -> None:
    data: Any = 1
    for _ in range(DEEP_NESTING_DEPTH):
        data = [data]
    assert to_flat_dict(data) == {".".join(["0"] * DEEP_NESTING_DEPTH): 1}


def test_to_flat_dict_does_not_modify_the_input() -> None:
    data = {"a": [1, {"b": 2}], "c": Node("n", [3])}
    to_flat_dict(data)
    assert data == {"a": [1, {"b": 2}], "c": Node("n", [3])}
