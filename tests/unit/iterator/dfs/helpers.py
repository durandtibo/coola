from __future__ import annotations

from collections import OrderedDict

import pytest


class CustomList(list):
    r"""Create a custom class that inherits from list."""


SAMPLES: pytest.mark.ParameterSet = [
    pytest.param("abc", ["abc"], id="str"),
    pytest.param(42, [42], id="int"),
    pytest.param("", [""], id="empty string"),
    # iterable
    pytest.param([5, 3, 8, 1, 9, 2], [5, 3, 8, 1, 9, 2], id="list"),
    pytest.param([], [], id="empty list"),
    pytest.param((1, 2, 3), [1, 2, 3], id="tuple"),
    pytest.param((), [], id="empty tuple"),
    pytest.param([[1, 2, 3], [4, 5, 6]], [1, 2, 3, 4, 5, 6], id="nested list"),
    pytest.param(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [1, 2, 3, 4, 5, 6, 7, 8], id="deeply nested"
    ),
    pytest.param([1, "a", 2.5, None, True], [1, "a", 2.5, None, True], id="mixed types"),
    pytest.param(range(3), [0, 1, 2], id="generator"),
    # mapping
    pytest.param({"a": 1, "b": 2}, [1, 2], id="dict"),
    pytest.param({"a": {"b": 1, "c": 2}, "d": 3}, [1, 2, 3], id="nested dict"),
    pytest.param({}, [], id="empty dict"),
    pytest.param({"a": {}}, [], id="empty nested dict"),
    pytest.param({"x": [1, 2], "y": [3, 4]}, [1, 2, 3, 4], id="nested dict list"),
    pytest.param({"a": {"b": [1, 2], "c": 3}, "d": 4}, [1, 2, 3, 4], id="nested dict mixed types"),
    pytest.param(OrderedDict({"a": 1, "b": 2}), [1, 2], id="ordered dict"),
]


DEFAULT_SAMPLES: pytest.mark.ParameterSet = [
    pytest.param("abc", ["abc"], id="str"),
    pytest.param(42, [42], id="int"),
    pytest.param({"key": "value"}, [{"key": "value"}], id="dict"),
    pytest.param({}, [{}], id="empty dict"),
    pytest.param([1, 2, 3], [[1, 2, 3]], id="list"),
    pytest.param([], [[]], id="empty list"),
    pytest.param((1, 2, 3, 4), [(1, 2, 3, 4)], id="tuple"),
    pytest.param((), [()], id="empty tuple"),
    pytest.param({1, 2, 3}, [{1, 2, 3}], id="set"),
    pytest.param(set(), [set()], id="empty set"),
]
