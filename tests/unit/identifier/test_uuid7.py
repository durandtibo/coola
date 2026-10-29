from __future__ import annotations

import re
import time
import uuid
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from coola.identifier import uuid7 as uuid7_module
from coola.identifier.uuid7 import extract_uuid7_timestamp_ms, generate_uuid7

if TYPE_CHECKING:
    from collections.abc import Generator

UUID7_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


@pytest.fixture(autouse=True)
def _reset_sequence_state() -> Generator[None]:
    """Reset the state of the sequence, which is shared by the tests of
    the process and depends on the time (a test with a fake clock must
    start from scratch)."""
    uuid7_module._SEQUENCE._reset_after_fork()
    yield
    uuid7_module._SEQUENCE._reset_after_fork()


def _decode_timestamp_ms(value: str) -> int:
    """Decode the 48-bit timestamp encoded in a UUIDv7 string.

    Used to verify the timestamp is actually round-tripped through the
    encoding correctly, rather than just checking the output's shape.
    """
    return int(value.replace("-", ""), 16) >> 80


####################################
#     Tests for generate_uuid7     #
####################################


def test_generate_uuid7_returns_str() -> None:
    assert isinstance(generate_uuid7(), str)


def test_generate_uuid7_length() -> None:
    assert len(generate_uuid7()) == 36


def test_generate_uuid7_matches_pattern() -> None:
    assert UUID7_PATTERN.match(generate_uuid7())


def test_generate_uuid7_is_unique() -> None:
    assert generate_uuid7() != generate_uuid7()


def test_generate_uuid7_sorts_by_timestamp() -> None:
    early = generate_uuid7(timestamp_ms=0)
    late = generate_uuid7(timestamp_ms=1_000_000)
    assert early < late


def test_generate_uuid7_same_timestamp_still_differs() -> None:
    # Randomness component should differ even for the same timestamp.
    assert generate_uuid7(timestamp_ms=42) != generate_uuid7(timestamp_ms=42)


def test_generate_uuid7_zero_timestamp() -> None:
    assert UUID7_PATTERN.match(generate_uuid7(timestamp_ms=0))


def test_generate_uuid7_max_timestamp() -> None:
    assert UUID7_PATTERN.match(generate_uuid7(timestamp_ms=2**48 - 1))


def test_generate_uuid7_negative_timestamp_raises() -> None:
    with pytest.raises(ValueError, match="timestamp_ms must be in the range"):
        generate_uuid7(timestamp_ms=-1)


def test_generate_uuid7_timestamp_too_large_raises() -> None:
    with pytest.raises(ValueError, match="timestamp_ms must be in the range"):
        generate_uuid7(timestamp_ms=2**48)


def test_generate_uuid7_encodes_timestamp_roundtrip() -> None:
    assert _decode_timestamp_ms(generate_uuid7(timestamp_ms=1_234_567_890_123)) == 1_234_567_890_123


def test_generate_uuid7_encodes_zero_timestamp_roundtrip() -> None:
    assert _decode_timestamp_ms(generate_uuid7(timestamp_ms=0)) == 0


def test_generate_uuid7_encodes_max_timestamp_roundtrip() -> None:
    assert _decode_timestamp_ms(generate_uuid7(timestamp_ms=2**48 - 1)) == 2**48 - 1


def test_generate_uuid7_default_timestamp_is_current_time() -> None:
    before = int(time.time() * 1000)
    uuid7 = generate_uuid7()
    after = int(time.time() * 1000)
    assert before <= _decode_timestamp_ms(uuid7) <= after


def test_generate_uuid7_many_calls_are_unique() -> None:
    uuids = {generate_uuid7() for _ in range(1000)}
    assert len(uuids) == 1000


################################################
#     Tests for extract_uuid7_timestamp_ms     #
################################################


def test_extract_uuid7_timestamp_ms_roundtrip() -> None:
    assert extract_uuid7_timestamp_ms(generate_uuid7(timestamp_ms=1_234_567_890_123)) == (
        1_234_567_890_123
    )


def test_extract_uuid7_timestamp_ms_zero() -> None:
    assert extract_uuid7_timestamp_ms(generate_uuid7(timestamp_ms=0)) == 0


def test_extract_uuid7_timestamp_ms_max() -> None:
    assert extract_uuid7_timestamp_ms(generate_uuid7(timestamp_ms=2**48 - 1)) == 2**48 - 1


def test_extract_uuid7_timestamp_ms_matches_decode_helper() -> None:
    uuid7 = generate_uuid7()
    assert extract_uuid7_timestamp_ms(uuid7) == _decode_timestamp_ms(uuid7)


def test_extract_uuid7_timestamp_ms_invalid_string_raises() -> None:
    with pytest.raises(ValueError, match="canonical 36-character form"):
        extract_uuid7_timestamp_ms("not-a-uuid")


def test_extract_uuid7_timestamp_ms_rejects_non_v7_uuid() -> None:
    # A UUIDv4-shaped string: version nibble is 4, not 7.
    with pytest.raises(ValueError, match="uuid7 must be a version 7 UUID"):
        extract_uuid7_timestamp_ms("00000000-0000-4000-8000-000000000000")


##########################################
#     Tests for the monotonic UUIDv7     #
##########################################


def _generate_uuid7_in_the_same_millisecond() -> list[str]:
    with patch("coola.identifier.sequence.time.time_ns", return_value=1_700_000_000_000 * 10**6):
        return [generate_uuid7() for _ in range(1000)]


def test_generate_uuid7_monotonic_in_the_same_millisecond() -> None:
    values = _generate_uuid7_in_the_same_millisecond()
    assert values == sorted(values)


def test_generate_uuid7_unique_in_the_same_millisecond() -> None:
    assert len(set(_generate_uuid7_in_the_same_millisecond())) == 1000


def test_generate_uuid7_same_timestamp_in_the_same_millisecond() -> None:
    values = _generate_uuid7_in_the_same_millisecond()
    assert {extract_uuid7_timestamp_ms(v) for v in values} == {1_700_000_000_000}


def test_generate_uuid7_valid_pattern_in_the_same_millisecond() -> None:
    values = _generate_uuid7_in_the_same_millisecond()
    assert all(UUID7_PATTERN.match(v) for v in values)


def test_generate_uuid7_monotonic_with_the_real_clock_sorted() -> None:
    values = [generate_uuid7() for _ in range(5000)]
    assert values == sorted(values)


def test_generate_uuid7_monotonic_with_the_real_clock_unique() -> None:
    values = [generate_uuid7() for _ in range(5000)]
    assert len(set(values)) == 5000


def test_generate_uuid7_monotonic_with_the_real_clock_pattern() -> None:
    values = [generate_uuid7() for _ in range(5000)]
    assert all(UUID7_PATTERN.match(v) for v in values)


def _generate_uuid7_with_a_backward_clock() -> tuple[str, str]:
    with patch("coola.identifier.sequence.time.time_ns", return_value=2_000 * 10**6):
        first = generate_uuid7()
    with patch("coola.identifier.sequence.time.time_ns", return_value=1_000 * 10**6):
        second = generate_uuid7()
    return first, second


def test_generate_uuid7_monotonic_if_the_clock_moves_backward() -> None:
    first, second = _generate_uuid7_with_a_backward_clock()
    assert second > first


def test_generate_uuid7_keeps_the_timestamp_if_the_clock_moves_backward() -> None:
    _, second = _generate_uuid7_with_a_backward_clock()
    assert extract_uuid7_timestamp_ms(second) == 2_000


def _generate_uuid7_with_a_large_counter() -> list[str]:
    # Start with a random part that is all ones in the 12 bits before the variant bits.
    with (
        patch("coola.identifier.sequence.time.time_ns", return_value=1_000 * 10**6),
        patch.object(uuid7_module._SEQUENCE, "_start", return_value=2**74 - 3),
    ):
        return [generate_uuid7() for _ in range(5)]


def test_generate_uuid7_monotonic_keeps_version_and_variant_when_the_counter_is_large() -> None:
    values = _generate_uuid7_with_a_large_counter()
    assert all(UUID7_PATTERN.match(v) for v in values)


def test_generate_uuid7_monotonic_when_the_counter_is_large() -> None:
    values = _generate_uuid7_with_a_large_counter()
    assert values == sorted(values)


def test_generate_uuid7_unique_when_the_counter_is_large() -> None:
    # the overflow borrowed the next millisecond
    assert len(set(_generate_uuid7_with_a_large_counter())) == 5


def test_generate_uuid7_explicit_timestamp_is_used() -> None:
    assert extract_uuid7_timestamp_ms(generate_uuid7(timestamp_ms=1)) == 1


def test_generate_uuid7_explicit_timestamp_is_before_the_current_time() -> None:
    with patch("coola.identifier.sequence.time.time_ns", return_value=5_000 * 10**6):
        current = generate_uuid7()
    assert generate_uuid7(timestamp_ms=1) < current


def test_generate_uuid7_explicit_timestamp_does_not_use_the_monotonic_state() -> None:
    with patch("coola.identifier.sequence.time.time_ns", return_value=5_000 * 10**6):
        current = generate_uuid7()
    generate_uuid7(timestamp_ms=1)
    with patch("coola.identifier.sequence.time.time_ns", return_value=5_000 * 10**6):
        assert generate_uuid7() > current


#####################################################
#     Tests for the strict inputs and inverse     #
#####################################################


@pytest.mark.parametrize("timestamp_ms", [True, 1.5, "1"])
def test_generate_uuid7_timestamp_wrong_type_raises(timestamp_ms: object) -> None:
    with pytest.raises(TypeError, match="timestamp_ms must be an int"):
        generate_uuid7(timestamp_ms=timestamp_ms)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("uuid7", [None, 123, uuid.UUID(int=7 << 76)])
def test_extract_uuid7_timestamp_ms_not_a_str_raises(uuid7: object) -> None:
    with pytest.raises(TypeError, match="uuid7 must be an instance of"):
        extract_uuid7_timestamp_ms(uuid7)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "uuid7",
    [
        "{01234567-89ab-7def-8123-456789abcdef}",
        "urn:uuid:01234567-89ab-7def-8123-456789abcdef",
        "0123456789ab7def8123456789abcdef",
        " 01234567-89ab-7def-8123-456789abcdef",
        "01234567-89ab-7def-8123-456789abcdef\n",
    ],
)
def test_extract_uuid7_timestamp_ms_not_canonical_raises(uuid7: str) -> None:
    with pytest.raises(ValueError, match="canonical 36-character form"):
        extract_uuid7_timestamp_ms(uuid7)


def test_extract_uuid7_timestamp_ms_wrong_version_raises() -> None:
    with pytest.raises(ValueError, match="must be a version 7 UUID, got version '4'"):
        extract_uuid7_timestamp_ms("01234567-89ab-4def-8123-456789abcdef")


def test_extract_uuid7_timestamp_ms_wrong_variant_raises() -> None:
    with pytest.raises(ValueError, match="RFC 9562 variant, got variant 'c'"):
        extract_uuid7_timestamp_ms("01234567-89ab-7def-c123-456789abcdef")


def test_extract_uuid7_timestamp_ms_accepts_upper_case() -> None:
    assert extract_uuid7_timestamp_ms("01234567-89AB-7DEF-8123-456789ABCDEF") == 0x0123456789AB


def test_generate_uuid7_current_time_beyond_48_bits_raises() -> None:
    with (
        patch("coola.identifier.sequence.time.time_ns", return_value=2**48 * 10**6),
        pytest.raises(ValueError, match="timestamp_ms must be in the range"),
    ):
        generate_uuid7()


def test_generate_uuid7_timestamp_is_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional argument"):
        generate_uuid7(1704067200000)  # ty: ignore[too-many-positional-arguments]
