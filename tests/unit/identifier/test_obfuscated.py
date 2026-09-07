from __future__ import annotations

import pytest

from coola.identifier import decode_obfuscated_id, generate_obfuscated_id


def test_generate_obfuscated_id_returns_str() -> None:
    assert isinstance(generate_obfuscated_id(1), str)


def test_generate_obfuscated_id_roundtrip_values() -> None:
    for number in (0, 1, 42, 1234, 2**32, 2**64 - 1):
        encoded = generate_obfuscated_id(number, salt="orders")
        assert decode_obfuscated_id(encoded, salt="orders") == number


def test_generate_obfuscated_id_different_salt_differs() -> None:
    assert generate_obfuscated_id(42, salt="a") != generate_obfuscated_id(42, salt="b")


def test_generate_obfuscated_id_decode_wrong_salt_gives_wrong_value() -> None:
    encoded = generate_obfuscated_id(42, salt="right")
    assert decode_obfuscated_id(encoded, salt="wrong") != 42


def test_generate_obfuscated_id_min_length() -> None:
    encoded = generate_obfuscated_id(0, salt="", min_length=10)
    assert len(encoded) == 10


def test_generate_obfuscated_id_negative_raises() -> None:
    with pytest.raises(ValueError, match="number must fit in 64 bits"):
        generate_obfuscated_id(-1)


def test_generate_obfuscated_id_too_large_raises() -> None:
    with pytest.raises(ValueError, match="number must fit in 64 bits"):
        generate_obfuscated_id(2**64)


def test_decode_obfuscated_id_invalid_character_raises() -> None:
    with pytest.raises(ValueError, match="encoded contains a character outside"):
        decode_obfuscated_id("not!valid")
