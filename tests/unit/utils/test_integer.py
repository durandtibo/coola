from __future__ import annotations

import sys

import pytest

from coola.utils.integer import int_repr


@pytest.mark.parametrize("value", [0, 42, -7, 2**64])
def test_int_repr_small(value: int) -> None:
    assert int_repr(value) == repr(value)


def test_int_repr_above_the_conversion_limit() -> None:
    assert int_repr(10**5000) == f"0x{10**5000:x}"
    assert int_repr(-(10**5000)) == f"0x{-(10**5000):x}"


def test_int_repr_at_the_conversion_limit() -> None:
    limit = sys.get_int_max_str_digits()
    below = 10 ** (limit - 1)
    assert int_repr(below) == repr(below)
    assert int_repr(10**limit).startswith("0x")


def test_int_repr_bool_and_int_subclass() -> None:
    class MyInt(int):
        pass

    assert int_repr(True) == "1"
    assert int_repr(MyInt(5)) == "5"
