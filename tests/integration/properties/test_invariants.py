r"""Property-based tests (Hypothesis) for core invariants."""

from __future__ import annotations

from hypothesis import given, strategies as st

from coola.equality import objects_are_equal
from coola.hashing import hash_object
from coola.recursive import recursive_apply

scalars = st.none() | st.booleans() | st.integers() | st.text() | st.binary()
floats = st.floats(allow_nan=False)
keys = st.text() | st.integers()
objects = st.recursive(
    scalars | floats,
    lambda children: (
        st.lists(children, max_size=4)
        | st.tuples(children, children)
        | st.dictionaries(keys, children, max_size=4)
    ),
    max_leaves=15,
)


@given(objects)
def test_objects_are_equal_is_reflexive(obj: object) -> None:
    assert objects_are_equal(obj, obj)


@given(objects, objects)
def test_objects_are_equal_is_symmetric(a: object, b: object) -> None:
    assert objects_are_equal(a, b) == objects_are_equal(b, a)


@given(st.recursive(scalars | floats, lambda c: st.lists(c, max_size=4), max_leaves=10))
def test_hash_object_agrees_with_equality(obj: object) -> None:
    assert hash_object(obj) == hash_object(recursive_apply(obj, lambda x: x))


@given(objects, objects)
def test_equal_objects_have_equal_hashes(a: object, b: object) -> None:
    if objects_are_equal(a, b):
        assert hash_object(a, ignore_unhashable=True) == hash_object(b, ignore_unhashable=True)


@given(objects)
def test_recursive_apply_identity_returns_equal_object(obj: object) -> None:
    assert objects_are_equal(recursive_apply(obj, lambda x: x), obj)
