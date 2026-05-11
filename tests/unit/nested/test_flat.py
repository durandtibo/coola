from __future__ import annotations

from unittest.mock import Mock

import pytest

from coola.equality import objects_are_equal
from coola.nested import to_flat_dict
from coola.testing.fixtures import numpy_available, torch_available
from coola.utils.imports import is_numpy_available, is_torch_available

if is_numpy_available():
    import numpy as np
else:  # pragma: no cover
    np = Mock()

if is_torch_available():
    import torch
else:  # pragma: no cover
    torch = Mock()


##################################
#     Tests for to_flat_dict     #
##################################


def test_to_flat_dict_empty_dict() -> None:
    assert to_flat_dict({}) == {}


def test_to_flat_dict_flat_dict() -> None:
    assert to_flat_dict({"bool": False, "float": 3.5, "int": 2, "str": "abc"}) == {
        "bool": False,
        "float": 3.5,
        "int": 2,
        "str": "abc",
    }


def test_to_flat_dict_nested_dict() -> None:
    assert to_flat_dict({"a": "a", "b": {"c": "c"}, "d": {"e": {"f": "f"}}}) == {
        "a": "a",
        "b.c": "c",
        "d.e.f": "f",
    }


def test_to_flat_dict_nested_dict_empty_child_omitted() -> None:
    # An empty nested dict contributes no keys to the flat output.
    assert to_flat_dict({"module": {}, "str": "abc"}) == {"str": "abc"}


def test_to_flat_dict_deeply_nested_dict() -> None:
    assert to_flat_dict(
        {
            "module": {
                "component_a": {"float": 3.5, "int": 2},
                "component_b": {"param_a": 1, "param_b": 2},
                "str": "abc",
            }
        }
    ) == {
        "module.component_a.float": 3.5,
        "module.component_a.int": 2,
        "module.component_b.param_a": 1,
        "module.component_b.param_b": 2,
        "module.str": "abc",
    }


def test_to_flat_dict_non_string_dict_keys_are_stringified() -> None:
    # Integer keys should be converted to strings, not left as ints.
    assert to_flat_dict({0: "zero", 1: {"nested": "val"}}) == {
        "0": "zero",
        "1.nested": "val",
    }


def test_to_flat_dict_bare_list() -> None:
    # A list passed as the top-level value (no prefix) should be flattened.
    assert to_flat_dict([2, "abc", True, 3.5]) == {
        "0": 2,
        "1": "abc",
        "2": True,
        "3": 3.5,
    }


def test_to_flat_dict_dict_with_list() -> None:
    assert to_flat_dict({"module": [2, "abc", True, 3.5], "str": "abc"}) == {
        "module.0": 2,
        "module.1": "abc",
        "module.2": True,
        "module.3": 3.5,
        "str": "abc",
    }


def test_to_flat_dict_dict_with_tuple() -> None:
    assert to_flat_dict({"module": (2, "abc", True, 3.5), "str": "abc"}) == {
        "module.0": 2,
        "module.1": "abc",
        "module.2": True,
        "module.3": 3.5,
        "str": "abc",
    }


def test_to_flat_dict_nested_list_and_dict() -> None:
    assert to_flat_dict({"module": [[1, 2, 3], {"bool": True}], "str": "abc"}) == {
        "module.0.0": 1,
        "module.0.1": 2,
        "module.0.2": 3,
        "module.1.bool": True,
        "str": "abc",
    }


def test_to_flat_dict_nested_tuple_and_dict() -> None:
    assert to_flat_dict({"module": ([1, 2, 3], {"bool": True}), "str": "abc"}) == {
        "module.0.0": 1,
        "module.0.1": 2,
        "module.0.2": 3,
        "module.1.bool": True,
        "str": "abc",
    }


def test_to_flat_dict_prefix_flat_dict() -> None:
    assert to_flat_dict({"a": 1, "b": 2}, prefix="root") == {
        "root.a": 1,
        "root.b": 2,
    }


def test_to_flat_dict_prefix_nested_dict() -> None:
    assert to_flat_dict({"a": {"b": 1}}, prefix="root") == {"root.a.b": 1}


def test_to_flat_dict_prefix_with_list() -> None:
    assert to_flat_dict([10, 20], prefix="items") == {"items.0": 10, "items.1": 20}


def test_to_flat_dict_prefix_scalar_leaf() -> None:
    # A scalar with an explicit prefix should produce a single-entry dict.
    assert to_flat_dict(42, prefix="answer") == {"answer": 42}


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
    result = to_flat_dict({"items": [1, 2]}, separator=separator)
    assert result == {f"items{separator}0": 1, f"items{separator}1": 2}


def test_to_flat_dict_to_str_single_type_list() -> None:
    result = to_flat_dict(
        {"module": [[1, 2, 3], {"bool": True}], "str": "abc"},
        to_str=list,
    )
    assert result == {"module": "[[1, 2, 3], {'bool': True}]", "str": "abc"}


def test_to_flat_dict_to_str_single_type_tuple() -> None:
    result = to_flat_dict(
        {"module": (2, "abc", True, 3.5), "str": "abc"},
        to_str=tuple,
    )
    assert result == {"module": "(2, 'abc', True, 3.5)", "str": "abc"}


def test_to_flat_dict_to_str_multiple_types() -> None:
    result = to_flat_dict(
        {"module1": (2, "abc", True, 3.5), "module2": [1, 2, 3], "str": "abc"},
        to_str=(list, tuple),
    )
    assert result == {
        "module1": "(2, 'abc', True, 3.5)",
        "module2": "[1, 2, 3]",
        "str": "abc",
    }


def test_to_flat_dict_to_str_with_prefix() -> None:
    # A to_str-matched value at the top level requires a prefix.
    result = to_flat_dict([1, 2, 3], prefix="items", to_str=list)
    assert result == {"items": "[1, 2, 3]"}


def test_to_flat_dict_to_str_none_is_noop() -> None:
    # Explicitly passing to_str=None should behave the same as the default.
    assert to_flat_dict({"a": [1, 2]}, to_str=None) == {"a.0": 1, "a.1": 2}


def test_to_flat_dict_scalar_without_prefix_raises() -> None:
    with pytest.raises(ValueError, match="None key"):
        to_flat_dict(42)


def test_to_flat_dict_to_str_match_without_prefix_raises() -> None:
    with pytest.raises(ValueError, match="None key"):
        to_flat_dict([1, 2, 3], to_str=list)


@torch_available
def test_to_flat_dict_tensor_passthrough() -> None:
    tensor = torch.ones(2, 3)
    result = to_flat_dict({"tensor": tensor})
    assert objects_are_equal(result, {"tensor": torch.ones(2, 3)})


@torch_available
def test_to_flat_dict_tensor_mixed_with_scalars() -> None:
    tensor = torch.zeros(3)
    result = to_flat_dict({"a": 1, "b": tensor})
    assert result["a"] == 1
    assert objects_are_equal(result["b"], tensor)


@numpy_available
def test_to_flat_dict_numpy_array_passthrough() -> None:
    array = np.zeros((2, 3))
    result = to_flat_dict({"array": array})
    assert objects_are_equal(result, {"array": np.zeros((2, 3))})


@numpy_available
def test_to_flat_dict_numpy_array_mixed_with_scalars() -> None:
    array = np.ones(4)
    result = to_flat_dict({"x": "hello", "y": array})
    assert result["x"] == "hello"
    assert objects_are_equal(result["y"], array)
