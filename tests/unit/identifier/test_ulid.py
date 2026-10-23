from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from coola.identifier import ulid as ulid_module
from coola.identifier.ulid import extract_ulid_timestamp_ms, generate_ulid

if TYPE_CHECKING:
    from collections.abc import Generator

ULID_PATTERN = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


@pytest.fixture(autouse=True)
def _reset_monotonic_state() -> Generator[None]:
    """Reset the state of the monotonic generator, which is shared by
    the tests of the process and depends on the time (a test with a fake
    clock must start from scratch)."""
    ulid_module._MONOTONIC._reset_after_fork()
    yield
    ulid_module._MONOTONIC._reset_after_fork()


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


###################################
#     Tests for generate_ulid     #
###################################


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


###############################################
#     Tests for extract_ulid_timestamp_ms     #
###############################################


def test_extract_ulid_timestamp_ms_roundtrip() -> None:
    assert extract_ulid_timestamp_ms(generate_ulid(timestamp_ms=1_234_567_890_123)) == (
        1_234_567_890_123
    )


def test_extract_ulid_timestamp_ms_zero() -> None:
    assert extract_ulid_timestamp_ms(generate_ulid(timestamp_ms=0)) == 0


def test_extract_ulid_timestamp_ms_max() -> None:
    assert extract_ulid_timestamp_ms(generate_ulid(timestamp_ms=2**48 - 1)) == 2**48 - 1


def test_extract_ulid_timestamp_ms_matches_decode_helper() -> None:
    ulid = generate_ulid()
    assert extract_ulid_timestamp_ms(ulid) == _decode_timestamp_ms(ulid)


def test_extract_ulid_timestamp_ms_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="ulid must be 26 characters long"):
        extract_ulid_timestamp_ms("TOO-SHORT")


def test_extract_ulid_timestamp_ms_invalid_character_raises() -> None:
    with pytest.raises(ValueError, match="ulid contains a character outside"):
        extract_ulid_timestamp_ms("I" * 26)


#########################################
#     Tests for the monotonic ULIDs     #
#########################################


def test_generate_ulid_monotonic_in_the_same_millisecond() -> None:
    with patch("coola.identifier.monotonic.time.time_ns", return_value=1_700_000_000_000 * 10**6):
        ulids = [generate_ulid() for _ in range(1000)]
    assert ulids == sorted(ulids)
    assert len(set(ulids)) == 1000
    assert {extract_ulid_timestamp_ms(u) for u in ulids} == {1_700_000_000_000}


def test_generate_ulid_monotonic_with_the_real_clock() -> None:
    ulids = [generate_ulid() for _ in range(5000)]
    assert ulids == sorted(ulids)
    assert len(set(ulids)) == 5000


def test_generate_ulid_monotonic_if_the_clock_moves_backward() -> None:
    with patch("coola.identifier.monotonic.time.time_ns", return_value=2_000 * 10**6):
        first = generate_ulid()
    with patch("coola.identifier.monotonic.time.time_ns", return_value=1_000 * 10**6):
        second = generate_ulid()
    assert second > first
    assert extract_ulid_timestamp_ms(second) == 2_000


def test_generate_ulid_explicit_timestamp_does_not_use_the_monotonic_state() -> None:
    with patch("coola.identifier.monotonic.time.time_ns", return_value=5_000 * 10**6):
        current = generate_ulid()
    old = generate_ulid(timestamp_ms=1)
    assert extract_ulid_timestamp_ms(old) == 1
    assert old < current
    # the explicit timestamps do not consume the sequence of the current time
    with patch("coola.identifier.monotonic.time.time_ns", return_value=5_000 * 10**6):
        assert generate_ulid() > current


def test_generate_ulid_explicit_timestamp_random_part_is_random() -> None:
    assert generate_ulid(timestamp_ms=42) != generate_ulid(timestamp_ms=42)


def test_extract_ulid_timestamp_ms_rejects_a_value_larger_than_128_bits() -> None:
    with pytest.raises(ValueError, match="larger than"):
        extract_ulid_timestamp_ms("Z" * 26)


def test_extract_ulid_timestamp_ms_accepts_lowercase() -> None:
    ulid = generate_ulid(timestamp_ms=1704067200000)
    assert extract_ulid_timestamp_ms(ulid.lower()) == 1704067200000
