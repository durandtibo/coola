r"""Define helpers shared by the tests."""

from __future__ import annotations

from collections import UserString, deque
from types import MappingProxyType
from typing import Any

import pytest

__all__ = ["CONTAINER_SAMPLES", "DEEP_NESTING_DEPTH", "LEAF_SAMPLES", "nest"]

# Deeper than the default recursion limit (1000), so a recursive implementation fails.
DEEP_NESTING_DEPTH = 5_000

# Samples of the native types, shared by the tests that check a property of every default entry
# of a registry. A leaf has no children, a container has some. ``pytest.param`` ids name the type.
LEAF_SAMPLES = [
    pytest.param("abc", id="str"),
    pytest.param(UserString("abc"), id="user_string"),
    pytest.param(1, id="int"),
    pytest.param(1.5, id="float"),
    pytest.param(1j, id="complex"),
    pytest.param(True, id="bool"),
]

CONTAINER_SAMPLES = [
    pytest.param([1, 2], id="list"),
    pytest.param((1, 2), id="tuple"),
    pytest.param(deque([1, 2], maxlen=3), id="deque"),
    pytest.param({1, 2}, id="set"),
    pytest.param(frozenset({1, 2}), id="frozenset"),
    pytest.param({"a": 1, "b": 2}, id="dict"),
    pytest.param(MappingProxyType({"a": 1}), id="mapping_proxy"),
]


def nest(leaf: Any, depth: int) -> Any:
    r"""Create deeply nested data.

    Args:
        leaf: The innermost value.
        depth: The number of lists that wrap ``leaf``. If ``0``, the
            leaf is returned unchanged.

    Returns:
        ``leaf`` wrapped in ``depth`` lists, for example
            ``nest(1, 2)`` returns ``[[1]]``.
    """
    data = leaf
    for _ in range(depth):
        data = [data]
    return data
