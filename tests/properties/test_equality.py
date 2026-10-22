r"""Property-based tests for the guarantees of ``objects_are_equal`` and
``objects_are_allclose``.

The guarantees are documented in ``docs/docs/uguide/equality.md``.
"""

from __future__ import annotations

import copy
import datetime as dt
import math
from collections import OrderedDict, deque
from typing import Any

import pytest
from hypothesis import assume, given, strategies as st

from coola.equality import objects_are_allclose, objects_are_equal
from coola.hashing import hash_object
from tests.properties.strategies import nan_values, rebuild, scalars, small_values, values

finite_floats = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)


###############################
#     Reflexive/symmetric     #
###############################


@given(data=values)
def test_equal_is_reflexive(data: Any) -> None:
    assert objects_are_equal(data, data)
    assert objects_are_equal(data, copy.deepcopy(data))


@given(data=small_values, other=small_values)
def test_equal_is_symmetric(data: Any, other: Any) -> None:
    assert objects_are_equal(data, other) == objects_are_equal(other, data)


@given(a=small_values, b=small_values, c=small_values)
def test_equal_is_transitive(a: Any, b: Any, c: Any) -> None:
    if objects_are_equal(a, b) and objects_are_equal(b, c):
        assert objects_are_equal(a, c)


#######################
#     Strict type     #
#######################


@given(x=st.integers(min_value=-(2**52), max_value=2**52))
def test_strict_type_int_float_and_bool_are_pairwise_different(x: int) -> None:
    assert not objects_are_equal(x, float(x))
    assert not objects_are_equal(float(x), x)
    assert not objects_are_equal(x, str(x))


@pytest.mark.parametrize("x", [0, 1])
def test_strict_type_bool_differs_from_int_and_float(x: int) -> None:
    assert not objects_are_equal(bool(x), x)
    assert not objects_are_equal(bool(x), float(x))


@given(items=st.lists(st.integers(), max_size=5))
def test_strict_type_containers_with_same_items_differ(items: list[int]) -> None:
    containers = [items, tuple(items), deque(items), set(items), frozenset(items)]
    for i, a in enumerate(containers):
        for j, b in enumerate(containers):
            assert objects_are_equal(a, b) == (i == j)


@given(s=st.text())
def test_strict_type_str_differs_from_bytes(s: str) -> None:
    assert not objects_are_equal(s, s.encode())


@given(x=st.integers())
def test_strict_type_subclass_differs_from_base_class(x: int) -> None:
    class MyInt(int):
        pass

    assert not objects_are_equal(MyInt(x), x)
    assert not objects_are_equal(x, MyInt(x))
    assert objects_are_equal(MyInt(x), MyInt(x))


@given(data=st.dictionaries(st.text(max_size=5), st.integers(), max_size=4))
def test_strict_type_dict_differs_from_ordered_dict(data: dict) -> None:
    assert not objects_are_equal(data, OrderedDict(data))


#####################
#     Recursive     #
#####################


@given(items=st.lists(st.integers(), max_size=5), extra=st.integers())
def test_recursive_different_lengths_are_different(items: list[int], extra: int) -> None:
    assert not objects_are_equal(items, [*items, extra])
    assert not objects_are_equal(tuple(items), (*items, extra))


@given(
    data=st.dictionaries(st.text(max_size=5), st.integers(), max_size=4),
    key=st.text(max_size=5),
)
def test_recursive_different_keys_are_different(data: dict, key: str) -> None:
    assume(key not in data)
    assert not objects_are_equal(data, {**data, key: 0})
    if data:
        first = next(iter(data))
        renamed = {key if k == first else k: v for k, v in data.items()}
        assert not objects_are_equal(data, renamed)


@given(x=values, y=values, depth=st.integers(1, 6))
def test_recursive_difference_is_found_at_any_depth(x: Any, y: Any, depth: int) -> None:
    assume(not objects_are_equal(x, y))
    left, right = x, y
    for _ in range(depth):
        left, right = [left, 1], [right, 1]
        left, right = {"k": left}, {"k": right}
    assert not objects_are_equal(left, right)


#######################
#     Order rules     #
#######################


@given(items=st.lists(st.integers(), unique=True, min_size=2, max_size=8), data=st.data())
def test_order_sequences_are_order_sensitive(items: list[int], data: st.DataObject) -> None:
    shuffled = data.draw(st.permutations(items))
    assume(shuffled != items)
    assert not objects_are_equal(items, shuffled)
    assert not objects_are_equal(tuple(items), tuple(shuffled))


@given(
    mapping=st.dictionaries(st.text(max_size=5), st.integers(), max_size=6),
    data=st.data(),
)
def test_order_dict_ignores_the_order_of_the_keys(mapping: dict, data: st.DataObject) -> None:
    keys = data.draw(st.permutations(list(mapping)))
    assert objects_are_equal(mapping, {k: mapping[k] for k in keys})


@given(
    mapping=st.dictionaries(st.text(max_size=5), st.integers(), max_size=6),
    data=st.data(),
)
def test_order_ordered_dict_compares_the_order_of_the_keys(
    mapping: dict, data: st.DataObject
) -> None:
    keys = list(mapping)
    shuffled = data.draw(st.permutations(keys))
    other = OrderedDict((k, mapping[k]) for k in shuffled)
    assert objects_are_equal(OrderedDict(mapping), OrderedDict(mapping))
    assert objects_are_equal(OrderedDict(mapping), other) == (shuffled == keys)


@given(
    mapping=st.dictionaries(st.text(max_size=5), st.integers(), min_size=2, max_size=6),
    data=st.data(),
)
def test_order_nested_ordered_dict_is_checked_at_any_depth(
    mapping: dict, data: st.DataObject
) -> None:
    keys = list(mapping)
    shuffled = data.draw(st.permutations(keys))
    assume(shuffled != keys)
    actual = {"a": [OrderedDict(mapping)]}
    expected = {"a": [OrderedDict((k, mapping[k]) for k in shuffled)]}
    assert not objects_are_equal(actual, expected)
    assert not objects_are_allclose(actual, expected)


@given(items=st.lists(st.integers(), unique=True, max_size=8), data=st.data())
def test_order_sets_ignore_the_iteration_order(items: list[int], data: st.DataObject) -> None:
    shuffled = data.draw(st.permutations(items))
    assert objects_are_equal(set(items), set(shuffled))
    assert objects_are_equal(frozenset(items), frozenset(shuffled))


###############################
#     NaN and signed zero     #
###############################


def test_nan_is_not_equal_to_nan() -> None:
    assert not objects_are_equal(float("nan"), float("nan"))
    assert not objects_are_equal([float("nan")], [float("nan")])


@given(x=scalars)
def test_nan_same_object_is_always_equal(x: Any) -> None:
    nan = float("nan")
    assert objects_are_equal(nan, nan)
    assert objects_are_equal([x, nan], [x, nan])


@given(data=values)
def test_nan_equal_nan_makes_nan_equal(data: Any) -> None:
    assert objects_are_equal([data, float("nan")], [data, float("nan")], equal_nan=True)
    assert objects_are_allclose({"a": float("nan")}, {"a": float("nan")}, equal_nan=True)


def _fresh_floats(data: Any) -> Any:
    r"""Copy ``data`` with new float objects (``deepcopy`` keeps them),
    so that a NaN is not equal to its copy by identity."""
    if isinstance(data, float):
        return float.fromhex(data.hex())
    if isinstance(data, complex):
        return complex(_fresh_floats(data.real), _fresh_floats(data.imag))
    if isinstance(data, dict):
        return {k: _fresh_floats(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_fresh_floats(v) for v in data)
    return data


@given(data=nan_values)
def test_equal_nan_makes_nested_data_equal_to_its_copy(data: Any) -> None:
    other = _fresh_floats(copy.deepcopy(data))
    assert objects_are_equal(data, other, equal_nan=True)
    assert objects_are_allclose(data, other, equal_nan=True)


@given(x=values)
def test_nan_equal_nan_does_not_hide_a_difference(x: Any) -> None:
    assert not objects_are_equal([x, float("nan")], [x, float("nan"), 1], equal_nan=True)


@given(atol=st.floats(0, 1), rtol=st.floats(0, 1))
def test_signed_zero_is_equal_to_zero(atol: float, rtol: float) -> None:
    assert objects_are_equal(0.0, -0.0)
    assert objects_are_equal([-0.0], [0.0])
    assert objects_are_allclose(0.0, -0.0, atol=atol, rtol=rtol)


######################
#     Time zones     #
######################


@given(
    value=st.datetimes(
        min_value=dt.datetime(2000, 1, 1),  # noqa: DTZ001
        max_value=dt.datetime(2100, 1, 1),  # noqa: DTZ001
    )
)
def test_time_zones_aware_datetimes_compare_by_instant(value: dt.datetime) -> None:
    utc = value.replace(tzinfo=dt.UTC)
    for hours in (-5, 1, 9):
        other = utc.astimezone(dt.timezone(dt.timedelta(hours=hours)))
        assert objects_are_equal(utc, other)
    shifted = utc.astimezone(dt.timezone(dt.timedelta(hours=1))).replace(tzinfo=dt.UTC)
    assert not objects_are_equal(utc, shifted)


@given(value=st.datetimes())
def test_time_zones_naive_never_equals_aware(value: dt.datetime) -> None:
    assert not objects_are_equal(value, value.replace(tzinfo=dt.UTC))
    assert not objects_are_equal(value.replace(tzinfo=dt.UTC), value)
    assert objects_are_equal(value, value)


#####################
#     Tolerance     #
#####################


@given(data=small_values, other=small_values)
def test_tolerance_zero_allclose_behaves_like_equal(data: Any, other: Any) -> None:
    assert objects_are_allclose(data, other, atol=0.0, rtol=0.0) == objects_are_equal(data, other)


@given(data=values)
def test_tolerance_allclose_is_reflexive(data: Any) -> None:
    assert objects_are_allclose(data, copy.deepcopy(data))


@given(
    expected=finite_floats,
    atol=st.floats(0, 10),
    rtol=st.floats(0, 1),
    delta=st.floats(-20, 20),
)
def test_tolerance_of_floats_follows_math_isclose(
    expected: float, atol: float, rtol: float, delta: float
) -> None:
    actual = expected + delta
    # The tolerance of two floats is the one of ``math.isclose``.
    assert objects_are_allclose(actual, expected, atol=atol, rtol=rtol) == math.isclose(
        actual, expected, rel_tol=rtol, abs_tol=atol
    )


@given(x=finite_floats, atol=st.floats(0, 10))
def test_tolerance_is_applied_inside_containers(x: float, atol: float) -> None:
    assert objects_are_allclose({"a": [x]}, {"a": [x + atol / 2]}, atol=atol, rtol=0.0)


@given(x=st.integers(-1000, 1000))
def test_tolerance_does_not_relax_the_type_check(x: int) -> None:
    assert not objects_are_allclose(x, float(x), atol=1.0)
    assert not objects_are_allclose([x], (x,), atol=1.0)


@given(s=st.text(), t=st.text())
def test_tolerance_is_not_used_for_other_values(s: str, t: str) -> None:
    assert objects_are_allclose(s, t, atol=1e6, rtol=1e6) == (s == t)


##############################
#     Unregistered types     #
##############################


class WithoutEq:
    r"""A type without ``__eq__``."""


class WithEq:
    r"""A type with ``__eq__``."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, WithEq) and self.value == other.value

    __hash__ = None  # type: ignore[assignment]


def test_unregistered_type_without_eq_is_equal_only_to_itself() -> None:
    obj = WithoutEq()
    assert objects_are_equal(obj, obj)
    assert objects_are_equal([obj], [obj])
    assert not objects_are_equal(WithoutEq(), WithoutEq())


@given(a=st.integers(), b=st.integers())
def test_unregistered_type_with_eq_uses_eq(a: int, b: int) -> None:
    assert objects_are_equal(WithEq(a), WithEq(b)) == (a == b)
    assert objects_are_equal([WithEq(a)], [WithEq(b)]) == (a == b)


@given(x=values)
def test_unregistered_type_never_raises(x: Any) -> None:
    assert isinstance(objects_are_equal(WithoutEq(), x), bool)
    assert isinstance(objects_are_equal(x, WithoutEq()), bool)


################
#     Pure     #
################


@given(data=values, other=values)
def test_pure_inputs_are_not_modified_and_answer_is_stable(data: Any, other: Any) -> None:
    data_before, other_before = copy.deepcopy(data), copy.deepcopy(other)
    first = objects_are_equal(data, other)
    assert objects_are_equal(data, other) == first
    assert objects_are_equal(data, data_before)
    assert objects_are_equal(other, other_before)


#########################
#     Failure modes     #
#########################


def test_failure_mode_self_referencing_lists_are_equal() -> None:
    a: list = []
    a.append(a)
    b: list = []
    b.append(b)
    assert objects_are_equal(a, b)
    assert objects_are_allclose(a, b)


@given(x=st.integers(), y=st.integers())
def test_failure_mode_cycles_are_compared_by_their_values(x: int, y: int) -> None:
    a: list = [x]
    a.append(a)
    b: list = [y]
    b.append(b)
    assert objects_are_equal(a, b) == (x == y)
    assert objects_are_equal(b, a) == (x == y)


@given(depth=st.integers(1, 5))
def test_failure_mode_cycles_of_different_lengths_are_equal(depth: int) -> None:
    # ``a = [a]`` and ``b = [[b]]`` unfold to the same infinite value.
    a: list = []
    a.append(a)
    b: list = []
    inner = b
    for _ in range(depth):
        new: list = []
        inner.append(new)
        inner = new
    inner.append(b)
    assert objects_are_equal(a, b)


@given(data=values, other=values)
def test_show_difference_does_not_change_the_result(data: Any, other: Any) -> None:
    assert objects_are_equal(data, other, show_difference=True) == objects_are_equal(data, other)


#################################
#     Consistent w/ hashing     #
#################################


@given(data=small_values, other=small_values)
def test_equal_objects_have_the_same_hash(data: Any, other: Any) -> None:
    if objects_are_equal(data, other):
        assert hash_object(data) == hash_object(other)


@given(data=values)
def test_equal_rebuilt_objects_have_the_same_hash(data: Any) -> None:
    other = rebuild(data)
    assert objects_are_equal(data, other)
    assert hash_object(data) == hash_object(other)
