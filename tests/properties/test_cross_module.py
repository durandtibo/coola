r"""Property-based tests for the invariants shared by several
modules."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from coola.equality import objects_are_equal
from coola.hashing import hash_object
from coola.summary import summarize
from tests.properties.strategies import (
    nan_values,
    rebuild,
    small_values,
    tiny_values,
    values,
)

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import (  # noqa: E402
    assume,
    given,
    strategies as st,
)


@given(data=values)
def test_hash_object_is_unchanged_by_deepcopy(data: Any) -> None:
    assert hash_object(data) == hash_object(copy.deepcopy(data))


@given(data=small_values)
def test_rebuilt_objects_are_equal(data: Any) -> None:
    assert objects_are_equal(data, rebuild(data))


@given(data=small_values)
def test_equal_objects_have_the_same_hash(data: Any) -> None:
    assert hash_object(data) == hash_object(rebuild(data))


@given(a=tiny_values, b=tiny_values)
def test_equal_implies_same_hash(a: Any, b: Any) -> None:
    assume(objects_are_equal(a, b))
    assert hash_object(a) == hash_object(b)


@given(data=nan_values)
def test_summarize_never_raises(data: Any) -> None:
    assert isinstance(summarize(data), str)


class _BadRepr:
    r"""An object whose ``__repr__`` and ``__str__`` raise."""

    def __repr__(self) -> str:
        msg = "broken __repr__"
        raise ZeroDivisionError(msg)

    __str__ = __repr__


@given(items=st.lists(st.just(_BadRepr()) | st.integers(), max_size=5), max_depth=st.integers(0, 2))
def test_summarize_never_raises_with_a_failing_repr(items: list[Any], max_depth: int) -> None:
    assert isinstance(summarize({"a": items, "b": tuple(items)}, max_depth=max_depth), str)
