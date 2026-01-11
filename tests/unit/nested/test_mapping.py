from __future__ import annotations

from unittest.mock import Mock

import pytest

from coola import objects_are_equal
from coola.nested import get_first_value, remove_keys_starting_with, to_flat_dict
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

#####################################
#     Tests for get_first_value     #
#####################################


def test_get_first_value_empty() -> None:
    with pytest.raises(
        ValueError, match=r"First value cannot be returned because the mapping is empty"
    ):
        get_first_value({})


def test_get_first_value() -> None:
    assert get_first_value({"key1": 1, "key2": 2}) == 1


##################################
#     Tests for to_flat_dict     #
##################################


def test_to_flat_dict_flat_dict() -> None:
    flatten_dict = to_flat_dict(
        {
            "bool": False,
            "float": 3.5,
            "int": 2,
            "str": "abc",
        }
    )
    assert flatten_dict == {
        "bool": False,
        "float": 3.5,
        "int": 2,
        "str": "abc",
    }


def test_to_flat_dict_nested_dict_str() -> None:
    flatten_dict = to_flat_dict({"a": "a", "b": {"c": "c"}, "d": {"e": {"f": "f"}}})
    assert flatten_dict == {"a": "a", "b.c": "c", "d.e.f": "f"}


def test_to_flat_dict_nested_dict_multiple_types() -> None:
    flatten_dict = to_flat_dict(
        {
            "module": {
                "bool": False,
                "float": 3.5,
                "int": 2,
            },
            "str": "abc",
        }
    )
    assert flatten_dict == {
        "module.bool": False,
        "module.float": 3.5,
        "module.int": 2,
        "str": "abc",
    }


def test_to_flat_dict_data_empty_key() -> None:
    flatten_dict = to_flat_dict(
        {
            "module": {},
            "str": "abc",
        }
    )
    assert flatten_dict == {"str": "abc"}


def test_to_flat_dict_double_data() -> None:
    flatten_dict = to_flat_dict(
        {
            "str": "def",
            "module": {
                "component": {
                    "float": 3.5,
                    "int": 2,
                },
            },
        }
    )
    assert flatten_dict == {
        "module.component.float": 3.5,
        "module.component.int": 2,
        "str": "def",
    }


def test_to_flat_dict_double_data_2() -> None:
    flatten_dict = to_flat_dict(
        {
            "module": {
                "component_a": {
                    "float": 3.5,
                    "int": 2,
                },
                "component_b": {
                    "param_a": 1,
                    "param_b": 2,
                },
                "str": "abc",
            },
        }
    )
    assert flatten_dict == {
        "module.component_a.float": 3.5,
        "module.component_a.int": 2,
        "module.component_b.param_a": 1,
        "module.component_b.param_b": 2,
        "module.str": "abc",
    }


def test_to_flat_dict_list() -> None:
    flatten_dict = to_flat_dict([2, "abc", True, 3.5])
    assert flatten_dict == {
        "0": 2,
        "1": "abc",
        "2": True,
        "3": 3.5,
    }


def test_to_flat_dict_dict_with_list() -> None:
    flatten_dict = to_flat_dict(
        {
            "module": [2, "abc", True, 3.5],
            "str": "abc",
        }
    )
    assert flatten_dict == {
        "module.0": 2,
        "module.1": "abc",
        "module.2": True,
        "module.3": 3.5,
        "str": "abc",
    }


def test_to_flat_dict_with_more_complex_list() -> None:
    flatten_dict = to_flat_dict(
        {
            "module": [[1, 2, 3], {"bool": True}],
            "str": "abc",
        }
    )
    assert flatten_dict == {
        "module.0.0": 1,
        "module.0.1": 2,
        "module.0.2": 3,
        "module.1.bool": True,
        "str": "abc",
    }


def test_to_flat_dict_tuple() -> None:
    flatten_dict = to_flat_dict(
        {
            "module": (2, "abc", True, 3.5),
            "str": "abc",
        }
    )
    assert flatten_dict == {
        "module.0": 2,
        "module.1": "abc",
        "module.2": True,
        "module.3": 3.5,
        "str": "abc",
    }


def test_to_flat_dict_with_complex_tuple() -> None:
    flatten_dict = to_flat_dict(
        {
            "module": ([1, 2, 3], {"bool": True}),
            "str": "abc",
        }
    )
    assert flatten_dict == {
        "module.0.0": 1,
        "module.0.1": 2,
        "module.0.2": 3,
        "module.1.bool": True,
        "str": "abc",
    }


@pytest.mark.parametrize("separator", [".", "/", "@", "[SEP]"])
def test_to_flat_dict_separator(separator: str) -> None:
    flatten_dict = to_flat_dict(
        {
            "str": "def",
            "module": {
                "component": {
                    "float": 3.5,
                    "int": 2,
                },
            },
        },
        separator=separator,
    )
    assert flatten_dict == {
        f"module{separator}component{separator}float": 3.5,
        f"module{separator}component{separator}int": 2,
        "str": "def",
    }


def test_to_flat_dict_to_str_tuple() -> None:
    flatten_dict = to_flat_dict(
        {
            "module": (2, "abc", True, 3.5),
            "str": "abc",
        },
        to_str=tuple,
    )
    assert flatten_dict == {
        "module": "(2, 'abc', True, 3.5)",
        "str": "abc",
    }


def test_to_flat_dict_to_str_tuple_and_list() -> None:
    flatten_dict = to_flat_dict(
        {
            "module1": (2, "abc", True, 3.5),
            "module2": [1, 2, 3],
            "str": "abc",
        },
        to_str=(list, tuple),
    )
    assert flatten_dict == {
        "module1": "(2, 'abc', True, 3.5)",
        "module2": "[1, 2, 3]",
        "str": "abc",
    }


@torch_available
def test_to_flat_dict_tensor() -> None:
    assert objects_are_equal(
        to_flat_dict({"tensor": torch.ones(2, 3)}), {"tensor": torch.ones(2, 3)}
    )


@numpy_available
def test_to_flat_dict_numpy_ndarray() -> None:
    assert objects_are_equal(to_flat_dict(np.zeros((2, 3))), {None: np.zeros((2, 3))})


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
