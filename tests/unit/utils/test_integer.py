from __future__ import annotations

import sys

import pytest

from coola.utils.integer import int_repr


@pytest.mark.parametrize("value", [0, 42, -7, 2**64])
def test_int_repr_small(value: int) -> None:
    assert int_repr(value) == repr(value)


def test_int_repr_above_the_conversion_limit() -> None:
    assert int_repr(10**5000) == f"0x{10**5000:x}"


def test_int_repr_above_the_conversion_limit_negative() -> None:
    assert int_repr(-(10**5000)) == hex(-(10**5000))


def test_int_repr_below_the_conversion_limit() -> None:
    value = 10**4299
    assert int_repr(value) == str(value)


def test_int_repr_at_the_conversion_limit() -> None:
    assert int_repr(10**4300).startswith("0x")


def test_int_repr_bool() -> None:
    assert int_repr(True) == "1"


def test_int_repr_int_subclass() -> None:
    class MyInt(int):
        pass

    assert int_repr(MyInt(5)) == "5"


def test_int_repr_does_not_depend_on_the_conversion_limit_above() -> None:
    expected = int_repr(10**5000)
    limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        assert int_repr(10**5000) == expected
    finally:
        sys.set_int_max_str_digits(limit)


def test_int_repr_does_not_depend_on_the_conversion_limit_below() -> None:
    expected = str(10**2000)
    limit = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(1000)
    try:
        assert int_repr(10**2000) == expected
    finally:
        sys.set_int_max_str_digits(limit)


def test_int_repr_largest_decimal_negative() -> None:
    assert int_repr(-(10**4299)) == f"-1{'0' * 4299}"


def test_int_repr_huge_negative_has_the_sign_before_the_prefix() -> None:
    assert int_repr(-(1 << 20000)).startswith("-0x1000")
