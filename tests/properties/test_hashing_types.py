r"""Property-based tests for the hashing of the numbers, the ranges and
the dataclasses."""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np
from hypothesis import assume, given, strategies as st

from coola.hashing import hash_object

if TYPE_CHECKING:
    from fractions import Fraction

decimals = st.decimals(allow_nan=False, allow_infinity=False, places=4)
fractions = st.fractions(min_value=-100, max_value=100, max_denominator=50)


@given(value=decimals, extra_zeros=st.integers(0, 5))
def test_decimals_with_extra_zeros_are_equal(value: Decimal, extra_zeros: int) -> None:
    other = Decimal(f"{value}{'0' * extra_zeros}") if "." in str(value) else value
    assert other == value


@given(value=decimals, extra_zeros=st.integers(0, 5))
def test_equal_decimals_have_the_same_hash(value: Decimal, extra_zeros: int) -> None:
    other = Decimal(f"{value}{'0' * extra_zeros}") if "." in str(value) else value
    assert hash_object(value) == hash_object(other)


@given(first=decimals, second=decimals)
def test_decimal_hash_is_consistent_with_equality(first: Decimal, second: Decimal) -> None:
    assert (hash_object(first) == hash_object(second)) == (first == second)


@given(first=fractions, second=fractions)
def test_fraction_hash_is_consistent_with_equality(first: Fraction, second: Fraction) -> None:
    assert (hash_object(first) == hash_object(second)) == (first == second)


@given(
    first=st.builds(range, st.integers(-20, 20), st.integers(-20, 20), st.integers(1, 5)),
    second=st.builds(range, st.integers(-20, 20), st.integers(-20, 20), st.integers(-5, -1)),
)
def test_range_hash_is_consistent_with_equality(first: range, second: range) -> None:
    assert (hash_object(first) == hash_object(second)) == (first == second)


@given(first=st.builds(range, st.integers(-20, 20), st.integers(-20, 20), st.integers(1, 5)))
def test_range_hash_is_consistent_with_equality_for_the_same_range(first: range) -> None:
    assert hash_object(first) == hash_object(first)


@given(value=st.floats(allow_nan=False, width=32), dtype=st.sampled_from([np.float32, np.float64]))
def test_numpy_float_hashes_like_the_builtin_float(value: float, dtype: type) -> None:
    scalar = dtype(value)
    assert hash_object(scalar) == hash_object(float(scalar))


@given(value=st.integers(-(2**31), 2**31 - 1), dtype=st.sampled_from([np.int32, np.int64]))
def test_numpy_int_hashes_like_the_builtin_int(value: int, dtype: type) -> None:
    assert hash_object(dtype(value)) == hash_object(value)


@given(first=st.integers(-1000, 1000), second=st.integers(-1000, 1000))
def test_numpy_int_hash_is_consistent_with_equality(first: int, second: int) -> None:
    assert (hash_object(np.int64(first)) == hash_object(np.int64(second))) == (first == second)


@dataclasses.dataclass
class Record:
    name: str
    values: list[int]
    extra: Any = None


@given(name=st.text(max_size=5), values=st.lists(st.integers(), max_size=4))
def test_dataclass_hash_is_stable(name: str, values: list[int]) -> None:
    assert hash_object(Record(name, values)) == hash_object(Record(name, list(values)))


@given(name=st.text(max_size=5), values=st.lists(st.integers(), max_size=4))
def test_dataclass_hash_uses_the_fields(name: str, values: list[int]) -> None:
    assert hash_object(Record(name, values)) != hash_object(Record(name, [*values, 0]))


@given(
    first=st.builds(Record, st.text(max_size=3), st.lists(st.integers(-3, 3), max_size=3)),
    second=st.builds(Record, st.text(max_size=3), st.lists(st.integers(-3, 3), max_size=3)),
)
def test_dataclass_hash_is_consistent_with_equality(first: Record, second: Record) -> None:
    assert (hash_object(first) == hash_object(second)) == (first == second)


@given(name=st.text(max_size=5))
def test_dataclass_nested_in_containers(name: str) -> None:
    data = {"records": [Record(name, [1]), (Record(name, [2]),)]}
    assert hash_object(data) == hash_object({"records": [Record(name, [1]), (Record(name, [2]),)]})


@given(name=st.text(max_size=5))
def test_dataclass_nested_in_containers_uses_the_fields(name: str) -> None:
    data = {"records": [Record(name, [1]), (Record(name, [2]),)]}
    assume(name != "other")
    assert hash_object(data) != hash_object(
        {"records": [Record("other", [1]), (Record(name, [2]),)]}
    )
