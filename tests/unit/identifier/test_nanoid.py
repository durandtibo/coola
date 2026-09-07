from __future__ import annotations

import pytest

from coola.identifier import generate_nano_id


def test_generate_nano_id_returns_str() -> None:
    assert isinstance(generate_nano_id(), str)


def test_generate_nano_id_default_length() -> None:
    assert len(generate_nano_id()) == 21


def test_generate_nano_id_custom_size() -> None:
    assert len(generate_nano_id(size=8)) == 8


def test_generate_nano_id_is_unique() -> None:
    assert generate_nano_id() != generate_nano_id()


def test_generate_nano_id_many_calls_are_unique() -> None:
    values = {generate_nano_id() for _ in range(1000)}
    assert len(values) == 1000


def test_generate_nano_id_custom_alphabet() -> None:
    value = generate_nano_id(size=100, alphabet="01")
    assert len(value) == 100
    assert set(value) <= {"0", "1"}


def test_generate_nano_id_zero_size_raises() -> None:
    with pytest.raises(ValueError, match="size must be positive"):
        generate_nano_id(size=0)


def test_generate_nano_id_negative_size_raises() -> None:
    with pytest.raises(ValueError, match="size must be positive"):
        generate_nano_id(size=-1)


def test_generate_nano_id_empty_alphabet_raises() -> None:
    with pytest.raises(ValueError, match="alphabet must not be empty"):
        generate_nano_id(alphabet="")


def test_generate_nano_id_duplicate_alphabet_raises() -> None:
    with pytest.raises(ValueError, match="alphabet must not contain duplicate characters"):
        generate_nano_id(alphabet="aab")


def test_generate_nano_id_alphabet_too_large_raises() -> None:
    with pytest.raises(ValueError, match="alphabet must have at most 256 distinct characters"):
        generate_nano_id(alphabet="".join(chr(i) for i in range(300)))


def test_generate_nano_id_single_character_alphabet() -> None:
    assert generate_nano_id(size=5, alphabet="a") == "aaaaa"
