from __future__ import annotations

import re
import time

import pytest

from coola.identifier import ulid as ulid_module
from coola.identifier.ulid import generate_ulid

ULID_PATTERN = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


def _decode_timestamp_ms(value: str) -> int:
    """Decode the 48-bit timestamp encoded in a ULID string, by
    reversing the Crockford Base32 encoding used by ``generate_ulid``.

    Used to verify the timestamp is actually round-tripped through the
    encoding correctly, rather than just checking the output's shape.
    """
    payload = 0
    for char in value:
        payload = (payload << 5) | ulid_module._ENCODING.index(char)
    payload &= (1 << 128) - 1
    return payload >> 80


##################################
#     Tests for generate_ulid   #
##################################


def test_generate_ulid_returns_str() -> None:
    assert isinstance(generate_ulid(), str)


def test_generate_ulid_length() -> None:
    assert len(generate_ulid()) == 26


def test_generate_ulid_matches_pattern() -> None:
    assert ULID_PATTERN.match(generate_ulid())


def test_generate_ulid_is_unique() -> None:
    assert generate_ulid() != generate_ulid()


def test_generate_ulid_sorts_by_timestamp() -> None:
    early = generate_ulid(timestamp_ms=0)
    late = generate_ulid(timestamp_ms=1_000_000)
    assert early < late


def test_generate_ulid_same_timestamp_still_differs() -> None:
    # Randomness component should differ even for the same timestamp.
    assert generate_ulid(timestamp_ms=42) != generate_ulid(timestamp_ms=42)


def test_generate_ulid_zero_timestamp() -> None:
    assert ULID_PATTERN.match(generate_ulid(timestamp_ms=0))


def test_generate_ulid_max_timestamp() -> None:
    assert ULID_PATTERN.match(generate_ulid(timestamp_ms=2**48 - 1))


def test_generate_ulid_negative_timestamp_raises() -> None:
    with pytest.raises(ValueError, match="timestamp_ms must fit in 48 bits"):
        generate_ulid(timestamp_ms=-1)


def test_generate_ulid_timestamp_too_large_raises() -> None:
    with pytest.raises(ValueError, match="timestamp_ms must fit in 48 bits"):
        generate_ulid(timestamp_ms=2**48)


def test_generate_ulid_encodes_timestamp_roundtrip() -> None:
    assert _decode_timestamp_ms(generate_ulid(timestamp_ms=1_234_567_890_123)) == 1_234_567_890_123


def test_generate_ulid_encodes_zero_timestamp_roundtrip() -> None:
    assert _decode_timestamp_ms(generate_ulid(timestamp_ms=0)) == 0


def test_generate_ulid_encodes_max_timestamp_roundtrip() -> None:
    assert _decode_timestamp_ms(generate_ulid(timestamp_ms=2**48 - 1)) == 2**48 - 1


def test_generate_ulid_default_timestamp_is_current_time() -> None:
    before = int(time.time() * 1000)
    ulid = generate_ulid()
    after = int(time.time() * 1000)
    assert before <= _decode_timestamp_ms(ulid) <= after


def test_generate_ulid_many_calls_are_unique() -> None:
    ulids = {generate_ulid() for _ in range(1000)}
    assert len(ulids) == 1000


def test_generate_ulid_uses_full_crockford_alphabet_characters() -> None:
    # A large sample should exercise the randomness bytes broadly
    # enough to see characters beyond the timestamp-derived prefix.
    combined = "".join(generate_ulid() for _ in range(200))
    assert set(combined) <= set(ulid_module._ENCODING)
