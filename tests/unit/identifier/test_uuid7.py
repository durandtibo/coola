from __future__ import annotations

import re
import time

import pytest

from coola.identifier.uuid7 import generate_uuid7

UUID7_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def _decode_timestamp_ms(value: str) -> int:
    """Decode the 48-bit timestamp encoded in a UUIDv7 string.

    Used to verify the timestamp is actually round-tripped through the
    encoding correctly, rather than just checking the output's shape.
    """
    return int(value.replace("-", ""), 16) >> 80


####################################
#     Tests for generate_uuid7    #
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
    with pytest.raises(ValueError, match="timestamp_ms must fit in 48 bits"):
        generate_uuid7(timestamp_ms=-1)


def test_generate_uuid7_timestamp_too_large_raises() -> None:
    with pytest.raises(ValueError, match="timestamp_ms must fit in 48 bits"):
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
