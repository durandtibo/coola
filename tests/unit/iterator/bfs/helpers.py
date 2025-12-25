from __future__ import annotations

from collections import OrderedDict

import pytest


class CustomList(list):
    r"""Create a custom class that inherits from list."""


DEFAULT_FIND_CHILDREN_SAMPLES: pytest.mark.ParameterSet = [
    pytest.param("abc", [], id="str"),
    pytest.param(bytes("abc", "utf-8"), [], id="bytes"),
    pytest.param(42, [], id="int"),
    pytest.param({"key": "value"}, [], id="dict"),
    pytest.param({}, [], id="empty dict"),
    pytest.param([1, 2, 3], [], id="list"),
    pytest.param([], [], id="empty list"),
    pytest.param((1, 2, 3, 4), [], id="tuple"),
    pytest.param((), [], id="empty tuple"),
    pytest.param({1, 2, 3}, [], id="set"),
    pytest.param(set(), [], id="empty set"),
    pytest.param(None, [], id="none"),
]

DEFAULT_ITERATE_SAMPLES: pytest.mark.ParameterSet = [
    pytest.param("abc", ["abc"], id="str"),
    pytest.param(bytes("abc", "utf-8"), [bytes("abc", "utf-8")], id="bytes"),
    pytest.param(42, [42], id="int"),
    pytest.param({"key": "value"}, [], id="dict"),
    pytest.param({}, [], id="empty dict"),
    pytest.param([1, 2, 3], [], id="list"),
    pytest.param([], [], id="empty list"),
    pytest.param((1, 2, 3, 4), [], id="tuple"),
    pytest.param((), [], id="empty tuple"),
    pytest.param({1, 2, 3}, [], id="set"),
    pytest.param(set(), [], id="empty set"),
    pytest.param(None, [None], id="none"),
]

ITERATE_SAMPLES: pytest.mark.ParameterSet = [
    pytest.param("abc", ["abc"], id="str"),
    pytest.param(bytes("abc", "utf-8"), [bytes("abc", "utf-8")], id="bytes"),
    pytest.param(42, [42], id="int"),
    pytest.param({"key": "value"}, ["value"], id="dict"),
    pytest.param({}, [], id="empty dict"),
    pytest.param([1, 2, 3], [1, 2, 3], id="list"),
    pytest.param([], [], id="empty list"),
    pytest.param((1, 2, 3, 4), [1, 2, 3, 4], id="tuple"),
    pytest.param((), [], id="empty tuple"),
    pytest.param({1, 2, 3}, [1, 2, 3], id="set"),
    pytest.param(set(), [], id="empty set"),
    pytest.param(None, [None], id="none"),
    pytest.param({"a": {"b": 1, "c": 2}, "d": 3}, [3, 1, 2], id="nested dict"),
    pytest.param({"x": [1, 2], "y": [3, 4]}, [1, 2, 3, 4], id="dict list"),
    pytest.param(OrderedDict({"a": 1, "b": 2}), [1, 2], id="ordered dict"),
    pytest.param(
        {"a": {"b": 2, "c": {"d": 1, "e": 4}}, "d": 3}, [3, 2, 1, 4], id="deeply nested dict"
    ),
    pytest.param(
        {"a": [1, 2, 3], "b": {"c": 4, "d": [5, 6]}, "e": 7},
        [7, 1, 2, 3, 4, 5, 6],
        id="deeply nested dict",
    ),
    pytest.param([[1, 2, 3], [4, 5, 6]], [1, 2, 3, 4, 5, 6], id="nested list"),
    pytest.param(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        [1, 2, 3, 4, 5, 6, 7, 8],
        id="deeply nested list",
    ),
    pytest.param(range(3), [0, 1, 2], id="range"),
]
