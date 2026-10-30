r"""Define helpers shared by the tests."""

from __future__ import annotations

import pickle
import random
from collections import UserString, deque
from types import MappingProxyType
from typing import Any

import pytest

from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np

__all__ = [
    "CONTAINER_SAMPLES",
    "DEEP_NESTING_DEPTH",
    "LEAF_SAMPLES",
    "draw_numpy_rand",
    "draw_numpy_randint",
    "draw_numpy_randn",
    "draw_python_random",
    "get_numpy_state",
    "make_python_generator",
    "nest",
    "pickle_round_trip",
    "seed_numpy",
]

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


def draw_python_random() -> float:
    r"""Draw a number from the global generator of the ``random`` module.

    The tests of the random managers use it to check the seed and the
    state of this generator. It is the only place that calls ``random``
    for a draw (the rule ``S311`` targets cryptographic uses, which a
    test is not).

    Returns:
        A float in ``[0.0, 1.0)``.
    """
    return random.random()  # noqa: S311


def make_python_generator(seed: int) -> random.Random:
    r"""Make a ``random.Random`` instance, independent of the global
    generator.

    Args:
        seed: The seed of the generator.

    Returns:
        The seeded generator.
    """
    return random.Random(seed)  # noqa: S311


def draw_numpy_rand(*shape: int) -> Any:
    r"""Draw uniform numbers from the global generator of
    ``numpy.random``.

    The tests of the random managers use it to check the seed and the
    state of this generator (the rule ``NPY002`` targets the legacy API,
    which is the one the managers seed).

    Args:
        *shape: The shape of the output. If empty, a float is returned.

    Returns:
        The numbers in ``[0.0, 1.0)``.
    """
    return np.random.rand(*shape)  # noqa: NPY002


def draw_numpy_randn(*shape: int) -> Any:
    r"""Draw standard normal numbers from the global generator of
    ``numpy.random``.

    Args:
        *shape: The shape of the output. If empty, a float is returned.

    Returns:
        The numbers.
    """
    return np.random.randn(*shape)  # noqa: NPY002


def draw_numpy_randint(low: int, high: int, size: int) -> Any:
    r"""Draw integers from the global generator of ``numpy.random``.

    Args:
        low: The lowest integer (inclusive).
        high: The highest integer (exclusive).
        size: The number of integers.

    Returns:
        The integers.
    """
    return np.random.randint(low, high, size=size)  # noqa: NPY002


def get_numpy_state() -> Any:
    r"""Get the state of the global generator of ``numpy.random``.

    Returns:
        The state.
    """
    return np.random.get_state()  # noqa: NPY002


def seed_numpy(seed: Any) -> None:
    r"""Seed the global generator of ``numpy.random``.

    Args:
        seed: The seed, as accepted by ``numpy.random.seed``.
    """
    np.random.seed(seed)  # noqa: NPY002


def pickle_round_trip(obj: Any) -> Any:
    r"""Pickle an object and load it back.

    The tests of the picklable objects use it to check what survives a
    round trip (the rule ``S301`` targets untrusted data, which an object
    created by the test is not).

    Args:
        obj: The object to pickle.

    Returns:
        The loaded copy of ``obj``.
    """
    return pickle.loads(pickle.dumps(obj))  # noqa: S301
