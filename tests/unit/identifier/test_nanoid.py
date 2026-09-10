from __future__ import annotations

import pytest

from coola.identifier import generate_nano_id


def test_generate_nano_id_returns_str() -> None:
    assert isinstance(generate_nano_id(), str)


def test_generate_nano_id_default_length() -> None:
    assert len(generate_nano_id()) == 21


def test_generate_nano_id_custom_length() -> None:
    assert len(generate_nano_id(length=8)) == 8


def test_generate_nano_id_is_unique() -> None:
    assert generate_nano_id() != generate_nano_id()


def test_generate_nano_id_many_calls_are_unique() -> None:
    values = {generate_nano_id() for _ in range(1000)}
    assert len(values) == 1000


def test_generate_nano_id_custom_alphabet() -> None:
    value = generate_nano_id(length=100, alphabet="01")
    assert len(value) == 100
    assert set(value) <= {"0", "1"}


def test_generate_nano_id_zero_length_raises() -> None:
    with pytest.raises(ValueError, match="length must be positive"):
        generate_nano_id(length=0)


def test_generate_nano_id_negative_length_raises() -> None:
    with pytest.raises(ValueError, match="length must be positive"):
        generate_nano_id(length=-1)


def test_generate_nano_id_max_length_is_valid() -> None:
    nano_id = generate_nano_id(length=1024)
    assert len(nano_id) == 1024


def test_generate_nano_id_length_above_max_raises() -> None:
    with pytest.raises(ValueError, match="length must be at most 1024, got 1025"):
        generate_nano_id(length=1025)


def test_generate_nano_id_very_large_length_raises() -> None:
    with pytest.raises(ValueError, match="length must be at most 1024"):
        generate_nano_id(length=10_000_000)


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
    assert generate_nano_id(length=5, alphabet="a") == "aaaaa"


def test_generate_nano_id_rejects_out_of_range_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    # alphabet="abc" has 3 characters, so the rejection mask is 0b11
    # (3): a byte masking to 3 must be rejected since it is out of
    # range, and an accepted byte that does not yet complete the
    # requested length must resume the inner loop rather than return
    # early.
    buffer = bytes([3, 0, 3, 1, 2])  # reject, accept 'a', reject, accept 'b', accept 'c'
    monkeypatch.setattr("coola.identifier.nanoid.os.urandom", lambda _size: buffer)
    assert generate_nano_id(length=3, alphabet="abc") == "abc"


def test_generate_nano_id_needs_multiple_urandom_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    # The first os.urandom call returns only out-of-range bytes, so no
    # character is accepted and the outer loop must call os.urandom a
    # second time to make progress.
    batches = iter([bytes([3, 3, 3]), bytes([0, 1, 2])])
    monkeypatch.setattr("coola.identifier.nanoid.os.urandom", lambda _size: next(batches))
    assert generate_nano_id(length=3, alphabet="abc") == "abc"


def test_generate_nano_id_alphabet_just_above_power_of_two() -> None:
    # alphabet length 129 sits just above the 128 power-of-two boundary,
    # so the rejection mask (255) accepts only ~50% of bytes: a
    # worst-case-ish rejection rate worth exercising explicitly.
    alphabet = "".join(chr(ord("!") + i) for i in range(129))
    value = generate_nano_id(length=50, alphabet=alphabet)
    assert len(value) == 50
    assert all(char in alphabet for char in value)
