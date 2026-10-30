r"""Property-based tests for the guarantees of ``objects_are_equal`` and
``objects_are_allclose``.

The guarantees are documented in ``docs/docs/uguide/equality.md``.
"""

from __future__ import annotations

import copy
import datetime as dt
import functools
from collections import OrderedDict, deque
from typing import Any

import pytest
from hypothesis import assume, given, strategies as st

from coola.equality import compare, objects_are_allclose, objects_are_equal
from coola.hashing import hash_object
from tests.properties.strategies import (
    nan_values,
    rebuild,
    scalars,
    small_values,
    tiny_values,
    values,
)

finite_floats = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)


###############################
#     Reflexive/symmetric     #
###############################


@given(data=values)
def test_equal_is_reflexive(data: Any) -> None:
    assert objects_are_equal(data, data)


@given(data=values)
def test_equal_is_reflexive_for_a_copy(data: Any) -> None:
    assert objects_are_equal(data, copy.deepcopy(data))


@given(data=small_values, other=small_values)
def test_equal_is_symmetric(data: Any, other: Any) -> None:
    assert objects_are_equal(data, other) == objects_are_equal(other, data)


@given(a=small_values)
def test_equal_is_transitive(a: Any) -> None:
    c = copy.deepcopy(rebuild(a))
    assert objects_are_equal(a, c)


#######################
#     Strict type     #
#######################


class MyInt(int):
    r"""A subclass of ``int``."""


@given(x=st.integers(min_value=-(2**52), max_value=2**52))
def test_strict_type_int_differs_from_float(x: int) -> None:
    assert not objects_are_equal(x, float(x))


@given(x=st.integers(min_value=-(2**52), max_value=2**52))
def test_strict_type_float_differs_from_int(x: int) -> None:
    assert not objects_are_equal(float(x), x)


@given(x=st.integers(min_value=-(2**52), max_value=2**52))
def test_strict_type_int_differs_from_str(x: int) -> None:
    assert not objects_are_equal(x, str(x))


@pytest.mark.parametrize("x", [0, 1])
def test_strict_type_bool_differs_from_int(x: int) -> None:
    assert not objects_are_equal(bool(x), x)


@pytest.mark.parametrize("x", [0, 1])
def test_strict_type_bool_differs_from_float(x: int) -> None:
    assert not objects_are_equal(bool(x), float(x))


CONTAINERS = [list, tuple, deque, set, frozenset]


@pytest.mark.parametrize(
    ("cls", "other_cls"), [(a, b) for a in CONTAINERS for b in CONTAINERS if a is not b]
)
@given(items=st.lists(st.integers(), max_size=5))
def test_strict_type_containers_with_same_items_differ(
    cls: type, other_cls: type, items: list[int]
) -> None:
    assert not objects_are_equal(cls(items), other_cls(items))


@pytest.mark.parametrize("cls", CONTAINERS)
@given(items=st.lists(st.integers(), max_size=5))
def test_strict_type_containers_with_same_items_and_type_are_equal(
    cls: type, items: list[int]
) -> None:
    assert objects_are_equal(cls(items), cls(items))


@given(s=st.text())
def test_strict_type_str_differs_from_bytes(s: str) -> None:
    assert not objects_are_equal(s, s.encode())


@given(x=st.integers())
def test_strict_type_subclass_differs_from_base_class(x: int) -> None:
    assert not objects_are_equal(MyInt(x), x)


@given(x=st.integers())
def test_strict_type_base_class_differs_from_subclass(x: int) -> None:
    assert not objects_are_equal(x, MyInt(x))


@given(x=st.integers())
def test_strict_type_subclass_is_equal_to_itself(x: int) -> None:
    assert objects_are_equal(MyInt(x), MyInt(x))


@given(data=st.dictionaries(st.text(max_size=5), st.integers(), max_size=4))
def test_strict_type_dict_differs_from_ordered_dict(data: dict) -> None:
    assert not objects_are_equal(data, OrderedDict(data))


#####################
#     Recursive     #
#####################


@given(items=st.lists(st.integers(), max_size=5), extra=st.integers())
def test_recursive_lists_of_different_lengths_are_different(items: list[int], extra: int) -> None:
    assert not objects_are_equal(items, [*items, extra])


@given(items=st.lists(st.integers(), max_size=5), extra=st.integers())
def test_recursive_tuples_of_different_lengths_are_different(items: list[int], extra: int) -> None:
    assert not objects_are_equal(tuple(items), (*items, extra))


@given(
    data=st.dictionaries(st.text(max_size=5), st.integers(), max_size=4),
    key=st.text(max_size=5),
)
def test_recursive_extra_key_is_different(data: dict, key: str) -> None:
    assume(key not in data)
    assert not objects_are_equal(data, {**data, key: 0})


@given(
    data=st.dictionaries(st.text(max_size=5), st.integers(), min_size=1, max_size=4),
    key=st.text(max_size=5),
)
def test_recursive_renamed_key_is_different(data: dict, key: str) -> None:
    assume(key not in data)
    first = next(iter(data))
    renamed = {key if k == first else k: v for k, v in data.items()}
    assert not objects_are_equal(data, renamed)


@given(x=values, y=values, depth=st.integers(1, 6))
def test_recursive_difference_is_found_at_any_depth(x: Any, y: Any, depth: int) -> None:
    assume(not objects_are_equal(x, y))
    left, right = functools.reduce(
        lambda pair, _: ({"k": [pair[0], 1]}, {"k": [pair[1], 1]}), range(depth), (x, y)
    )
    assert not objects_are_equal(left, right)


#######################
#     Order rules     #
#######################


@given(items=st.lists(st.integers(), unique=True, min_size=2, max_size=8), data=st.data())
def test_order_lists_are_order_sensitive(items: list[int], data: st.DataObject) -> None:
    shuffled = data.draw(st.permutations(items))
    assume(shuffled != items)
    assert not objects_are_equal(items, shuffled)


@given(items=st.lists(st.integers(), unique=True, min_size=2, max_size=8), data=st.data())
def test_order_tuples_are_order_sensitive(items: list[int], data: st.DataObject) -> None:
    shuffled = data.draw(st.permutations(items))
    assume(shuffled != items)
    assert not objects_are_equal(tuple(items), tuple(shuffled))


@given(
    mapping=st.dictionaries(st.text(max_size=5), st.integers(), max_size=6),
    data=st.data(),
)
def test_order_dict_ignores_the_order_of_the_keys(mapping: dict, data: st.DataObject) -> None:
    keys = data.draw(st.permutations(list(mapping)))
    assert objects_are_equal(mapping, {k: mapping[k] for k in keys})


@given(mapping=st.dictionaries(st.text(max_size=5), st.integers(), max_size=6))
def test_order_ordered_dict_is_equal_to_itself(mapping: dict) -> None:
    assert objects_are_equal(OrderedDict(mapping), OrderedDict(mapping))


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


@given(
    mapping=st.dictionaries(st.text(max_size=5), st.integers(), min_size=2, max_size=6),
    data=st.data(),
)
def test_order_nested_ordered_dict_is_checked_at_any_depth_by_allclose(
    mapping: dict, data: st.DataObject
) -> None:
    keys = list(mapping)
    shuffled = data.draw(st.permutations(keys))
    assume(shuffled != keys)
    actual = {"a": [OrderedDict(mapping)]}
    expected = {"a": [OrderedDict((k, mapping[k]) for k in shuffled)]}
    assert not objects_are_allclose(actual, expected)


@given(items=st.lists(st.integers(), unique=True, max_size=8), data=st.data())
def test_order_sets_ignore_the_iteration_order(items: list[int], data: st.DataObject) -> None:
    shuffled = data.draw(st.permutations(items))
    assert objects_are_equal(set(items), set(shuffled))


@given(items=st.lists(st.integers(), unique=True, max_size=8), data=st.data())
def test_order_frozensets_ignore_the_iteration_order(items: list[int], data: st.DataObject) -> None:
    shuffled = data.draw(st.permutations(items))
    assert objects_are_equal(frozenset(items), frozenset(shuffled))


###############################
#     NaN and signed zero     #
###############################


def test_nan_is_not_equal_to_nan() -> None:
    assert not objects_are_equal(float("nan"), float("nan"))


def test_nan_in_a_list_is_not_equal_to_nan() -> None:
    assert not objects_are_equal([float("nan")], [float("nan")])


def test_nan_same_object_is_always_equal() -> None:
    nan = float("nan")
    assert objects_are_equal(nan, nan)


@given(x=scalars)
def test_nan_same_object_in_a_list_is_always_equal(x: Any) -> None:
    nan = float("nan")
    assert objects_are_equal([x, nan], [x, nan])


@given(data=values)
def test_nan_equal_nan_makes_nan_equal(data: Any) -> None:
    assert objects_are_equal([data, float("nan")], [data, float("nan")], equal_nan=True)


def test_nan_equal_nan_makes_nan_close() -> None:
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
    assert objects_are_equal(data, _fresh_floats(copy.deepcopy(data)), equal_nan=True)


@given(data=nan_values)
def test_equal_nan_makes_nested_data_close_to_its_copy(data: Any) -> None:
    assert objects_are_allclose(data, _fresh_floats(copy.deepcopy(data)), equal_nan=True)


@given(x=values)
def test_nan_equal_nan_does_not_hide_a_difference(x: Any) -> None:
    assert not objects_are_equal([x, float("nan")], [x, float("nan"), 1], equal_nan=True)


@given(atol=st.floats(0, 1), rtol=st.floats(0, 1))
def test_signed_zero_is_close_to_zero(atol: float, rtol: float) -> None:
    assert objects_are_allclose(0.0, -0.0, atol=atol, rtol=rtol)


def test_signed_zero_is_equal_to_zero() -> None:
    assert objects_are_equal(0.0, -0.0)


def test_signed_zero_in_a_list_is_equal_to_zero() -> None:
    assert objects_are_equal([-0.0], [0.0])


######################
#     Time zones     #
######################


@pytest.mark.parametrize("hours", [-5, 1, 9])
@given(
    value=st.datetimes(
        min_value=dt.datetime(2000, 1, 1, tzinfo=dt.UTC),
        max_value=dt.datetime(2100, 1, 1, tzinfo=dt.UTC),
    )
)
def test_time_zones_aware_datetimes_compare_by_instant(hours: int, value: dt.datetime) -> None:
    utc = value.replace(tzinfo=dt.UTC)
    other = utc.astimezone(dt.timezone(dt.timedelta(hours=hours)))
    assert objects_are_equal(utc, other)


@given(
    value=st.datetimes(
        min_value=dt.datetime(2000, 1, 1, tzinfo=dt.UTC),
        max_value=dt.datetime(2100, 1, 1, tzinfo=dt.UTC),
    )
)
def test_time_zones_aware_datetimes_with_a_shifted_instant_differ(value: dt.datetime) -> None:
    utc = value.replace(tzinfo=dt.UTC)
    shifted = utc.astimezone(dt.timezone(dt.timedelta(hours=1))).replace(tzinfo=dt.UTC)
    assert not objects_are_equal(utc, shifted)


@given(value=st.datetimes())
def test_time_zones_naive_never_equals_aware(value: dt.datetime) -> None:
    assert not objects_are_equal(value, value.replace(tzinfo=dt.UTC))


@given(value=st.datetimes())
def test_time_zones_aware_never_equals_naive(value: dt.datetime) -> None:
    assert not objects_are_equal(value.replace(tzinfo=dt.UTC), value)


@given(value=st.datetimes())
def test_time_zones_naive_equals_itself(value: dt.datetime) -> None:
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
def test_tolerance_of_floats_follows_the_numpy_rule(
    expected: float, atol: float, rtol: float, delta: float
) -> None:
    actual = expected + delta
    # The tolerance of two floats is the one of the arrays (symmetric NumPy rule).
    assert objects_are_allclose(actual, expected, atol=atol, rtol=rtol) == (
        abs(actual - expected) <= atol + rtol * max(abs(actual), abs(expected))
    )


@given(x=finite_floats, atol=st.floats(0, 10))
def test_tolerance_is_applied_inside_containers(x: float, atol: float) -> None:
    assert objects_are_allclose({"a": [x]}, {"a": [x + atol / 2]}, atol=atol, rtol=0.0)


@given(x=st.integers(-1000, 1000))
def test_tolerance_does_not_relax_the_type_check_of_scalars(x: int) -> None:
    assert not objects_are_allclose(x, float(x), atol=1.0)


@given(x=st.integers(-1000, 1000))
def test_tolerance_does_not_relax_the_type_check_of_containers(x: int) -> None:
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


def test_unregistered_type_without_eq_is_equal_to_itself() -> None:
    obj = WithoutEq()
    assert objects_are_equal(obj, obj)


def test_unregistered_type_without_eq_in_a_list_is_equal_to_itself() -> None:
    obj = WithoutEq()
    assert objects_are_equal([obj], [obj])


def test_unregistered_type_without_eq_is_different_from_another_instance() -> None:
    assert not objects_are_equal(WithoutEq(), WithoutEq())


@given(a=st.integers(), b=st.integers())
def test_unregistered_type_with_eq_uses_eq(a: int, b: int) -> None:
    assert objects_are_equal(WithEq(a), WithEq(b)) == (a == b)


@given(a=st.integers(), b=st.integers())
def test_unregistered_type_with_eq_in_a_list_uses_eq(a: int, b: int) -> None:
    assert objects_are_equal([WithEq(a)], [WithEq(b)]) == (a == b)


@given(x=values)
def test_unregistered_type_never_raises(x: Any) -> None:
    assert isinstance(objects_are_equal(WithoutEq(), x), bool)


@given(x=values)
def test_unregistered_type_as_expected_never_raises(x: Any) -> None:
    assert isinstance(objects_are_equal(x, WithoutEq()), bool)


################
#     Pure     #
################


@given(data=values, other=values)
def test_pure_answer_is_stable(data: Any, other: Any) -> None:
    first = objects_are_equal(data, other)
    assert objects_are_equal(data, other) == first


@given(data=values, other=values)
def test_pure_first_input_is_not_modified(data: Any, other: Any) -> None:
    data_before = copy.deepcopy(data)
    objects_are_equal(data, other)
    assert objects_are_equal(data, data_before)


@given(data=values, other=values)
def test_pure_second_input_is_not_modified(data: Any, other: Any) -> None:
    other_before = copy.deepcopy(other)
    objects_are_equal(data, other)
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


def test_failure_mode_self_referencing_lists_are_close() -> None:
    a: list = []
    a.append(a)
    b: list = []
    b.append(b)
    assert objects_are_allclose(a, b)


@given(x=st.integers(), y=st.integers())
def test_failure_mode_cycles_are_compared_by_their_values(x: int, y: int) -> None:
    a: list = [x]
    a.append(a)
    b: list = [y]
    b.append(b)
    assert objects_are_equal(a, b) == (x == y)


@given(x=st.integers(), y=st.integers())
def test_failure_mode_cycles_are_compared_by_their_values_reversed(x: int, y: int) -> None:
    a: list = [x]
    a.append(a)
    b: list = [y]
    b.append(b)
    assert objects_are_equal(b, a) == (x == y)


def _make_cycle(depth: int) -> list:
    r"""Create a list that contains itself after ``depth`` levels."""
    b: list = []
    inner = b
    for _ in range(depth):
        new: list = []
        inner.append(new)
        inner = new
    inner.append(b)
    return b


@given(depth=st.integers(1, 5))
def test_failure_mode_cycles_of_different_lengths_are_equal(depth: int) -> None:
    # ``a = [a]`` and ``b = [[b]]`` unfold to the same infinite value.
    a: list = []
    a.append(a)
    assert objects_are_equal(a, _make_cycle(depth))


@given(data=values, other=values)
def test_compare_agrees_with_objects_are_equal(data: Any, other: Any) -> None:
    assert bool(compare(data, other)) == objects_are_equal(data, other)


#################################
#     Consistent w/ hashing     #
#################################


@given(data=tiny_values, other=tiny_values)
def test_equal_objects_have_the_same_hash(data: Any, other: Any) -> None:
    assume(objects_are_equal(data, other))
    assert hash_object(data) == hash_object(other)


@given(data=values)
def test_rebuilt_objects_are_equal(data: Any) -> None:
    assert objects_are_equal(data, rebuild(data))


@given(data=values)
def test_equal_rebuilt_objects_have_the_same_hash(data: Any) -> None:
    assert hash_object(data) == hash_object(rebuild(data))


#################################################
#     Symmetry and monotonicity of allclose     #
#################################################


@given(x=finite_floats, y=finite_floats, atol=st.floats(0, 10), rtol=st.floats(0, 1))
def test_allclose_is_symmetric(x: float, y: float, atol: float, rtol: float) -> None:
    assert objects_are_allclose(x, y, atol=atol, rtol=rtol) == objects_are_allclose(
        y, x, atol=atol, rtol=rtol
    )


@given(
    x=finite_floats,
    delta=st.floats(-5, 5),
    atol=st.floats(0, 10),
    rtol=st.floats(0, 1),
    extra=st.floats(0, 10),
)
def test_allclose_is_monotonic_in_atol(
    x: float, delta: float, atol: float, rtol: float, extra: float
) -> None:
    data, other = [x], [x + delta]
    assume(objects_are_allclose(data, other, atol=atol, rtol=rtol))
    assert objects_are_allclose(data, other, atol=atol + extra, rtol=rtol)


@given(
    x=finite_floats,
    delta=st.floats(-5, 5),
    atol=st.floats(0, 10),
    rtol=st.floats(0, 1),
    extra=st.floats(0, 10),
)
def test_allclose_is_monotonic_in_rtol(
    x: float, delta: float, atol: float, rtol: float, extra: float
) -> None:
    data, other = [x], [x + delta]
    assume(objects_are_allclose(data, other, atol=atol, rtol=rtol))
    assert objects_are_allclose(data, other, atol=atol, rtol=rtol + extra)


###########################
#     Monotonicity        #
###########################

tolerances = st.floats(min_value=0.0, max_value=1.0)


@given(
    x=finite_floats,
    y=finite_floats,
    atol=tolerances,
    extra=tolerances,
)
def test_allclose_larger_atol_keeps_close_floats_close(
    x: float, y: float, atol: float, extra: float
) -> None:
    assert not objects_are_allclose(x, y, atol=atol, rtol=0.0) or objects_are_allclose(
        x, y, atol=atol + extra, rtol=0.0
    )


@given(
    x=finite_floats,
    y=finite_floats,
    rtol=tolerances,
    extra=tolerances,
)
def test_allclose_larger_rtol_keeps_close_floats_close(
    x: float, y: float, rtol: float, extra: float
) -> None:
    assert not objects_are_allclose(x, y, atol=0.0, rtol=rtol) or objects_are_allclose(
        x, y, atol=0.0, rtol=rtol + extra
    )


@given(
    x=st.lists(finite_floats, max_size=5),
    y=st.lists(finite_floats, max_size=5),
    atol=tolerances,
    extra=tolerances,
)
def test_allclose_larger_atol_keeps_close_lists_close(
    x: list[float], y: list[float], atol: float, extra: float
) -> None:
    assert not objects_are_allclose(x, y, atol=atol, rtol=0.0) or objects_are_allclose(
        x, y, atol=atol + extra, rtol=0.0
    )
