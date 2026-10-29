from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from coola.identifier import crockford, ulid as ulid_module
from coola.identifier.ulid import extract_ulid_timestamp_ms, generate_ulid

if TYPE_CHECKING:
    from collections.abc import Generator

ULID_PATTERN = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


@pytest.fixture(autouse=True)
def _reset_sequence_state() -> Generator[None]:
    """Reset the state of the sequence, which is shared by the tests of
    the process and depends on the time (a test with a fake clock must
    start from scratch)."""
    ulid_module._SEQUENCE._reset_after_fork()
    yield
    ulid_module._SEQUENCE._reset_after_fork()


def _decode_timestamp_ms(value: str) -> int:
    """Decode the 48-bit timestamp encoded in a ULID string, by
    reversing the Crockford Base32 encoding used by ``generate_ulid``.

    Used to verify the timestamp is actually round-tripped through the
    encoding correctly, rather than just checking the output's shape.
    """
    payload = 0
    for char in value:
        payload = (payload << 5) | crockford.ALPHABET.index(char)
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
    with pytest.raises(ValueError, match="timestamp_ms must be in the range"):
        generate_ulid(timestamp_ms=-1)


def test_generate_ulid_timestamp_too_large_raises() -> None:
    with pytest.raises(ValueError, match="timestamp_ms must be in the range"):
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
    assert set(combined) <= set(crockford.ALPHABET)


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
        extract_ulid_timestamp_ms("U" * 26)


#########################################
#     Tests for the monotonic ULIDs     #
#########################################


def _generate_ulids_in_the_same_millisecond() -> list[str]:
    with patch("coola.identifier.sequence.time.time_ns", return_value=1_700_000_000_000 * 10**6):
        return [generate_ulid() for _ in range(1000)]


def test_generate_ulid_monotonic_in_the_same_millisecond() -> None:
    ulids = _generate_ulids_in_the_same_millisecond()
    assert ulids == sorted(ulids)


def test_generate_ulid_unique_in_the_same_millisecond() -> None:
    assert len(set(_generate_ulids_in_the_same_millisecond())) == 1000


def test_generate_ulid_same_timestamp_in_the_same_millisecond() -> None:
    ulids = _generate_ulids_in_the_same_millisecond()
    assert {extract_ulid_timestamp_ms(u) for u in ulids} == {1_700_000_000_000}


def test_generate_ulid_monotonic_with_the_real_clock_sorted() -> None:
    ulids = [generate_ulid() for _ in range(5000)]
    assert ulids == sorted(ulids)


def test_generate_ulid_monotonic_with_the_real_clock_unique() -> None:
    ulids = [generate_ulid() for _ in range(5000)]
    assert len(set(ulids)) == 5000


def _generate_ulids_with_a_backward_clock() -> tuple[str, str]:
    with patch("coola.identifier.sequence.time.time_ns", return_value=2_000 * 10**6):
        first = generate_ulid()
    with patch("coola.identifier.sequence.time.time_ns", return_value=1_000 * 10**6):
        second = generate_ulid()
    return first, second


def test_generate_ulid_monotonic_if_the_clock_moves_backward() -> None:
    first, second = _generate_ulids_with_a_backward_clock()
    assert second > first


def test_generate_ulid_keeps_the_timestamp_if_the_clock_moves_backward() -> None:
    _, second = _generate_ulids_with_a_backward_clock()
    assert extract_ulid_timestamp_ms(second) == 2_000


def test_generate_ulid_explicit_timestamp_is_used() -> None:
    assert extract_ulid_timestamp_ms(generate_ulid(timestamp_ms=1)) == 1


def test_generate_ulid_explicit_timestamp_is_before_the_current_time() -> None:
    with patch("coola.identifier.sequence.time.time_ns", return_value=5_000 * 10**6):
        current = generate_ulid()
    assert generate_ulid(timestamp_ms=1) < current


def test_generate_ulid_explicit_timestamp_does_not_use_the_monotonic_state() -> None:
    # the explicit timestamps do not consume the sequence of the current time
    with patch("coola.identifier.sequence.time.time_ns", return_value=5_000 * 10**6):
        current = generate_ulid()
    generate_ulid(timestamp_ms=1)
    with patch("coola.identifier.sequence.time.time_ns", return_value=5_000 * 10**6):
        assert generate_ulid() > current


def test_generate_ulid_explicit_timestamp_random_part_is_random() -> None:
    assert generate_ulid(timestamp_ms=42) != generate_ulid(timestamp_ms=42)


def test_extract_ulid_timestamp_ms_rejects_a_value_larger_than_128_bits() -> None:
    with pytest.raises(ValueError, match="larger than"):
        extract_ulid_timestamp_ms("Z" * 26)


def test_extract_ulid_timestamp_ms_accepts_lowercase() -> None:
    ulid = generate_ulid(timestamp_ms=1704067200000)
    assert extract_ulid_timestamp_ms(ulid.lower()) == 1704067200000


#####################################################
#     Tests for the strict inputs and inverse     #
#####################################################


@pytest.mark.parametrize("timestamp_ms", [True, 1.5, "1"])
def test_generate_ulid_timestamp_wrong_type_raises(timestamp_ms: object) -> None:
    with pytest.raises(TypeError, match="timestamp_ms must be an int"):
        generate_ulid(timestamp_ms=timestamp_ms)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("ulid", [None, 123, b"0" * 26])
def test_extract_ulid_timestamp_ms_not_a_str_raises(ulid: object) -> None:
    with pytest.raises(TypeError, match="ulid must be an instance of"):
        extract_ulid_timestamp_ms(ulid)  # ty: ignore[invalid-argument-type]


def test_extract_ulid_timestamp_ms_reads_o_as_zero() -> None:
    assert extract_ulid_timestamp_ms("O" * 26) == 0


def test_extract_ulid_timestamp_ms_reads_i_and_l_as_one() -> None:
    assert extract_ulid_timestamp_ms("0000000001" + "I" * 16) == extract_ulid_timestamp_ms(
        "0000000001" + "L" * 16
    )


def test_extract_ulid_timestamp_ms_whitespace_raises() -> None:
    with pytest.raises(ValueError, match="ulid contains a character outside"):
        extract_ulid_timestamp_ms(" " + "0" * 25)


def test_extract_ulid_timestamp_ms_first_character_above_seven_raises() -> None:
    with pytest.raises(ValueError, match="the first character must be at most '7'"):
        extract_ulid_timestamp_ms("8" + "0" * 25)


def test_generate_ulid_current_time_beyond_48_bits_raises() -> None:
    with (
        patch("coola.identifier.sequence.time.time_ns", return_value=2**48 * 10**6),
        pytest.raises(ValueError, match="timestamp_ms must be in the range"),
    ):
        generate_ulid()


def test_generate_ulid_timestamp_is_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional argument"):
        generate_ulid(1704067200000)  # ty: ignore[too-many-positional-arguments]
