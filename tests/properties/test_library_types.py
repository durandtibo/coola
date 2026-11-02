r"""Property-based tests of the invariants of ``equality``, ``hashing``
and ``summary`` on the classes of values that the nested-container
strategies do not generate: signed zeros and NaNs, structured dtypes,
data frames with named indexes and categoricals, dataclasses with
``compare=False`` fields and items whose ``__repr__`` raises.

The tests marked ``xfail`` document a defect whose fix is not merged yet
(the item of the codebase review is in the reason).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, strategies as st

from coola.equality import objects_are_equal
from coola.hashing import hash_object
from coola.summary import summarize

floats = st.sampled_from([0.0, -0.0, 1.0, -1.0, float("nan"), -float("nan"), float("inf")])
# A list of floats and, for each item, whether the sign of the copy is flipped: a zero or a
# NaN with the other sign is equal to the original (with ``equal_nan=True``).
float_pairs = st.lists(st.tuples(floats, st.booleans()), min_size=1, max_size=3)
field_specs = st.tuples(st.sampled_from(["a", "b"]), st.sampled_from(["f8", "i8", "i4"]))
index_names = st.sampled_from([None, "x", "y"])
categories = st.sampled_from([["a", "b"], ["a", "b", "c"], ["b", "a"]])
frame_specs = st.tuples(index_names, index_names, categories, st.booleans())


def make_structured(spec: tuple[str, str], size: int = 2) -> np.ndarray:
    return np.zeros(size, dtype=[spec])


def make_frame(spec: tuple[Any, ...]) -> pd.DataFrame:
    index_name, columns_name, cats, ordered = spec
    return pd.DataFrame(
        {"c": pd.Categorical(["a", "b"], categories=cats, ordered=ordered)},
        index=pd.Index([0, 1], name=index_name),
    ).rename_axis(columns=columns_name)


@dataclasses.dataclass
class WithCache:
    value: int
    cache: int = dataclasses.field(default=0, compare=False)


class BadRepr:
    def __repr__(self) -> str:
        msg = "broken __repr__"
        raise ZeroDivisionError(msg)

    __str__ = __repr__


@pytest.mark.xfail(reason="NaN bit patterns (14.3)", strict=False)
@given(pairs=float_pairs)
def test_equal_float_arrays_have_the_same_hash(pairs: list[tuple[float, bool]]) -> None:
    first = np.array([x for x, _ in pairs])
    second = np.array([-x if flip else x for x, flip in pairs])
    assert objects_are_equal(first, second, equal_nan=True)
    assert hash_object(first) == hash_object(second)


@given(spec=field_specs, size=st.integers(1, 2))
def test_equal_structured_arrays_have_the_same_hash(spec: tuple[str, str], size: int) -> None:
    assert hash_object(make_structured(spec, size)) == hash_object(make_structured(spec, size))


@pytest.mark.xfail(reason="structured dtypes (14.1)", strict=False)
@given(first=field_specs, second=field_specs)
def test_different_structured_arrays_have_different_hashes(
    first: tuple[str, str], second: tuple[str, str]
) -> None:
    assume(first != second)
    assert not objects_are_equal(make_structured(first), make_structured(second))
    assert hash_object(make_structured(first)) != hash_object(make_structured(second))


@given(spec=frame_specs)
def test_equal_frames_have_the_same_hash(spec: tuple[Any, ...]) -> None:
    assert hash_object(make_frame(spec)) == hash_object(make_frame(spec))


@pytest.mark.xfail(reason="pandas metadata (14.2)", strict=False)
@given(first=frame_specs, second=frame_specs)
def test_different_frames_have_different_hashes(
    first: tuple[Any, ...], second: tuple[Any, ...]
) -> None:
    assume(first != second)
    assert not objects_are_equal(make_frame(first), make_frame(second))
    assert hash_object(make_frame(first)) != hash_object(make_frame(second))


@given(value=st.integers(0, 2), first=st.integers(0, 2), second=st.integers(0, 2))
def test_equal_dataclasses_have_the_same_hash(value: int, first: int, second: int) -> None:
    left, right = WithCache(value, first), WithCache(value, second)
    assume(objects_are_equal(left, right))
    assert hash_object(left) == hash_object(right)


@given(
    items=st.lists(st.just(BadRepr()) | st.integers(), max_size=4),
    max_depth=st.integers(0, 3),
)
def test_summarize_never_raises_with_failing_items(items: list[Any], max_depth: int) -> None:
    data = {"list": items, "tuple": tuple(items), "dict": dict(enumerate(items))}
    assert isinstance(summarize(data, max_depth=max_depth), str)


@given(spec=frame_specs)
def test_frames_are_summarized(spec: tuple[Any, ...]) -> None:
    assert isinstance(summarize(make_frame(spec)), str)


@given(pairs=float_pairs)
def test_float_arrays_are_equal_to_a_copy(pairs: list[tuple[float, bool]]) -> None:
    array = np.array([x for x, _ in pairs])
    assert objects_are_equal(array, array.copy(), equal_nan=True)
