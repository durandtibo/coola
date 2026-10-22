r"""Property-based tests for the invariants shared by several
modules."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from coola.equality import objects_are_equal
from coola.hashing import hash_object
from coola.nested import convert_to_jsonable
from coola.summary import summarize
from tests.properties.strategies import nan_values, nested, rebuild, small_values, values

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import (  # noqa: E402
    given,
    strategies as st,
)

# The documented limits of ``convert_to_jsonable`` (NaN and infinity are kept, complex numbers
# and non-string keys are not converted) are outside of this domain.
json_values = nested(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.text(),
        st.floats(allow_nan=False, allow_infinity=False),
    )
)


@given(data=values)
def test_hash_object_is_unchanged_by_deepcopy(data: Any) -> None:
    assert hash_object(data) == hash_object(copy.deepcopy(data))


@given(data=small_values)
def test_equal_objects_have_the_same_hash(data: Any) -> None:
    other = rebuild(data)
    assert objects_are_equal(data, other)
    assert hash_object(data) == hash_object(other)


@given(a=small_values, b=small_values)
def test_equal_implies_same_hash(a: Any, b: Any) -> None:
    if objects_are_equal(a, b):
        assert hash_object(a) == hash_object(b)


@given(data=nan_values)
def test_summarize_never_raises(data: Any) -> None:
    assert isinstance(summarize(data), str)


@given(data=json_values)
def test_convert_to_jsonable_is_accepted_by_strict_json(data: Any) -> None:
    json.dumps(convert_to_jsonable(data), allow_nan=False)
