r"""Property-based tests for the validators of ``coola.validation``."""

from __future__ import annotations

from functools import partial
from typing import Any

import pytest
from hypothesis import assume, given, strategies as st

from coola.validation import (
    validate_finite,
    validate_ge,
    validate_gt,
    validate_int,
    validate_isinstance,
    validate_le,
    validate_lt,
    validate_range,
)

VALIDATORS: dict[type, Any] = {
    bool: partial(validate_isinstance, cls=bool),
    int: validate_int,
    float: partial(validate_isinstance, cls=float),
    str: partial(validate_isinstance, cls=str),
    list: partial(validate_isinstance, cls=list),
    tuple: partial(validate_isinstance, cls=tuple),
    dict: partial(validate_isinstance, cls=dict),
    set: partial(validate_isinstance, cls=set),
    frozenset: partial(validate_isinstance, cls=frozenset),
}

samples = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(),
    st.text(),
    st.binary(),
    st.lists(st.integers(), max_size=3),
    st.tuples(st.integers()),
    st.dictionaries(st.text(max_size=2), st.integers(), max_size=3),
    st.sets(st.integers(), max_size=3),
    st.frozensets(st.integers(), max_size=3),
)


def _accepts(cls: type, value: object) -> bool:
    # ``bool`` is a subclass of ``int`` but ``validate_int`` rejects it.
    if cls is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, cls)


def _raises(validator: Any, value: object, **kwargs: Any) -> bool:
    try:
        validator(value, **kwargs)
    except TypeError:
        return True
    return False


@pytest.mark.parametrize("cls", list(VALIDATORS))
@given(value=samples)
def test_validator_accepts_exactly_the_expected_type(cls: type, value: object) -> None:
    assert _raises(VALIDATORS[cls], value) == (not _accepts(cls, value))


@pytest.mark.parametrize("cls", list(VALIDATORS))
@given(value=samples, name=st.text(alphabet="abcxyz_", min_size=1, max_size=8))
def test_validator_error_message_contains_the_name(cls: type, value: object, name: str) -> None:
    assume(not _accepts(cls, value))
    with pytest.raises(TypeError, match=f"^{name} must be"):
        VALIDATORS[cls](value, name=name)


@given(value=st.booleans())
def test_validate_int_rejects_bool(value: bool) -> None:
    assert _raises(validate_int, value)


@given(value=st.booleans())
def test_validate_isinstance_accepts_bool_for_int(value: bool) -> None:
    assert not _raises(validate_isinstance, value, cls=int)


@given(value=samples)
def test_at_most_one_builtin_validator_accepts_a_value(value: object) -> None:
    accepted = [cls for cls in VALIDATORS if _accepts(cls, value)]
    assert len(accepted) <= 1


@given(items=st.lists(st.integers(), max_size=5))
def test_validate_isinstance_list_accepts_list(items: list[int]) -> None:
    assert not _raises(VALIDATORS[list], items)


@pytest.mark.parametrize(
    ("convert", "validator"),
    [
        (tuple, VALIDATORS[list]),
        (set, VALIDATORS[list]),
        (list, VALIDATORS[tuple]),
        (list, VALIDATORS[set]),
        (set, VALIDATORS[frozenset]),
        (frozenset, VALIDATORS[set]),
    ],
)
@given(items=st.lists(st.integers(), max_size=5))
def test_container_validators_do_not_convert_between_types(
    convert: type, validator: Any, items: list[int]
) -> None:
    assert _raises(validator, convert(items))


################################################
#     Tests for the value validators     #
################################################

finite_floats = st.floats(allow_nan=False, allow_infinity=False)


@given(x=finite_floats, y=finite_floats)
def test_validate_ge_raises_exactly_when_lower(x: float, y: float) -> None:
    assert _raises_value(validate_ge, x, y) == (x < y)


@given(x=finite_floats, y=finite_floats)
def test_validate_gt_raises_exactly_when_lower_or_equal(x: float, y: float) -> None:
    assert _raises_value(validate_gt, x, y) == (x <= y)


@given(x=finite_floats, y=finite_floats)
def test_validate_le_raises_exactly_when_greater(x: float, y: float) -> None:
    assert _raises_value(validate_le, x, y) == (x > y)


@given(x=finite_floats, y=finite_floats)
def test_validate_lt_raises_exactly_when_greater_or_equal(x: float, y: float) -> None:
    assert _raises_value(validate_lt, x, y) == (x >= y)


@given(
    x=finite_floats,
    bounds=st.tuples(finite_floats, finite_floats).map(sorted),
    low_inclusive=st.booleans(),
    high_inclusive=st.booleans(),
)
def test_validate_range_is_the_conjunction_of_two_bound_checks(
    x: float, bounds: list[float], low_inclusive: bool, high_inclusive: bool
) -> None:
    low, high = bounds
    assume(low < high)
    lower = validate_ge if low_inclusive else validate_gt
    upper = validate_le if high_inclusive else validate_lt
    expected = _raises_value(lower, x, low) or _raises_value(upper, x, high)
    assert (
        _raises_value(
            validate_range,
            x,
            low,
            high,
            low_inclusive=low_inclusive,
            high_inclusive=high_inclusive,
        )
        == expected
    )


@given(x=finite_floats, bounds=st.tuples(finite_floats, finite_floats).map(sorted))
def test_validate_range_closed_matches_the_interval(x: float, bounds: list[float]) -> None:
    low, high = bounds
    assert _raises_value(validate_range, x, low, high) == (not low <= x <= high)


@given(
    x=finite_floats,
    bounds=st.tuples(finite_floats, finite_floats).map(sorted),
)
def test_validate_range_open_matches_the_interval(x: float, bounds: list[float]) -> None:
    low, high = bounds
    assume(low < high)
    raises = _raises_value(validate_range, x, low, high, low_inclusive=False, high_inclusive=False)
    assert raises == (not low < x < high)


@given(x=finite_floats, bounds=st.tuples(finite_floats, finite_floats).map(sorted))
def test_validate_range_half_open_matches_the_interval(x: float, bounds: list[float]) -> None:
    low, high = bounds
    assume(low < high)
    assert _raises_value(validate_range, x, low, high, high_inclusive=False) == (
        not low <= x < high
    )


@given(x=finite_floats)
def test_validate_finite_accepts_every_finite_float(x: float) -> None:
    validate_finite(x)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_validate_finite_rejects_non_finite_floats(value: float) -> None:
    assert _raises_value(validate_finite, value)


@given(y=finite_floats)
def test_validate_ge_rejects_nan(y: float) -> None:
    assert _raises_value(validate_ge, float("nan"), y)


@given(y=finite_floats)
def test_validate_le_rejects_nan(y: float) -> None:
    assert _raises_value(validate_le, float("nan"), y)


def _raises_value(validator: Any, *args: Any, **kwargs: Any) -> bool:
    try:
        validator(*args, **kwargs)
    except ValueError:
        return True
    return False
