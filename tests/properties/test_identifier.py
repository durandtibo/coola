r"""Property-based tests for the obfuscated identifiers."""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from coola.identifier import decode_obfuscated_id, generate_obfuscated_id
from coola.identifier.obfuscated import _MASK, _decode_base62, _encode_base62

numbers = st.integers(0, 2**64 - 1)
salts = st.text(max_size=8)


@given(number=numbers, salt=salts, min_length=st.integers(0, 30))
def test_obfuscated_id_round_trip(number: int, salt: str, min_length: int) -> None:
    encoded = generate_obfuscated_id(number, salt=salt, min_length=min_length)
    assert decode_obfuscated_id(encoded, salt=salt) == number


@given(number=numbers, salt=salts, min_length=st.integers(0, 30))
def test_obfuscated_id_has_at_least_the_minimum_length(
    number: int, salt: str, min_length: int
) -> None:
    assert len(generate_obfuscated_id(number, salt=salt, min_length=min_length)) >= min_length


@given(number=numbers, other=numbers, salt=salts)
def test_obfuscated_id_is_injective(number: int, other: int, salt: str) -> None:
    same = generate_obfuscated_id(number, salt=salt) == generate_obfuscated_id(other, salt=salt)
    assert same == (number == other)


@given(value=st.integers(_MASK + 1, 2**200), salt=salts)
def test_decode_rejects_the_values_out_of_range(value: int, salt: str) -> None:
    with pytest.raises(ValueError, match="does not encode a 64-bit value"):
        decode_obfuscated_id(_encode_base62(value, min_length=0), salt=salt)


@given(value=st.integers(0, _MASK))
def test_base62_round_trip(value: int) -> None:
    assert _decode_base62(_encode_base62(value, min_length=0)) == value
