from __future__ import annotations

import pytest


class CustomList(list):
    r"""Create a custom class that inherits from list."""


SAMPLES: pytest.mark.ParameterSet = []


DEFAULT_FIND_CHILDREN_SAMPLES: pytest.mark.ParameterSet = [
    pytest.param("abc", [], id="str"),
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
    pytest.param(42, [42], id="int"),
    pytest.param({"key": "value"}, [{"key": "value"}], id="dict"),
    pytest.param({}, [{}], id="empty dict"),
    pytest.param([1, 2, 3], [[1, 2, 3]], id="list"),
    pytest.param([], [[]], id="empty list"),
    pytest.param((1, 2, 3, 4), [(1, 2, 3, 4)], id="tuple"),
    pytest.param((), [()], id="empty tuple"),
    pytest.param({1, 2, 3}, [{1, 2, 3}], id="set"),
    pytest.param(set(), [set()], id="empty set"),
    pytest.param(None, [None], id="none"),
]

ITERATE_SAMPLES: pytest.mark.ParameterSet = [
    pytest.param("abc", ["abc"], id="str"),
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
]
