from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from coola.identifier import uuid7 as uuid7_module
from coola.identifier.uuid7 import extract_uuid7_timestamp_ms, generate_uuid7

if TYPE_CHECKING:
    from collections.abc import Generator

UUID7_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


@pytest.fixture(autouse=True)
def _reset_monotonic_state() -> Generator[None]:
    """Reset the state of the monotonic generator, which is shared by
    the tests of the process and depends on the time (a test with a fake
    clock must start from scratch)."""
    uuid7_module._MONOTONIC._reset_after_fork()
    yield
    uuid7_module._MONOTONIC._reset_after_fork()


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


###################################################
#     Tests for extract_uuid7_timestamp_ms       #
###################################################


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
    with pytest.raises(ValueError, match="badly formed hexadecimal UUID string"):
        extract_uuid7_timestamp_ms("not-a-uuid")


def test_extract_uuid7_timestamp_ms_rejects_non_v7_uuid() -> None:
    # A UUIDv4-shaped string: version nibble is 4, not 7.
    with pytest.raises(ValueError, match="uuid7 must be a version 7 UUID"):
        extract_uuid7_timestamp_ms("00000000-0000-4000-8000-000000000000")


############################################
#     Tests for the monotonic UUIDv7       #
############################################


def test_generate_uuid7_monotonic_in_the_same_millisecond() -> None:
    with patch("coola.identifier.monotonic.time.time_ns", return_value=1_700_000_000_000 * 10**6):
        values = [generate_uuid7() for _ in range(1000)]
    assert values == sorted(values)
    assert len(set(values)) == 1000
    assert {extract_uuid7_timestamp_ms(v) for v in values} == {1_700_000_000_000}
    assert all(UUID7_PATTERN.match(v) for v in values)


def test_generate_uuid7_monotonic_with_the_real_clock() -> None:
    values = [generate_uuid7() for _ in range(5000)]
    assert values == sorted(values)
    assert len(set(values)) == 5000
    assert all(UUID7_PATTERN.match(v) for v in values)


def test_generate_uuid7_monotonic_if_the_clock_moves_backward() -> None:
    with patch("coola.identifier.monotonic.time.time_ns", return_value=2_000 * 10**6):
        first = generate_uuid7()
    with patch("coola.identifier.monotonic.time.time_ns", return_value=1_000 * 10**6):
        second = generate_uuid7()
    assert second > first
    assert extract_uuid7_timestamp_ms(second) == 2_000


def test_generate_uuid7_monotonic_keeps_version_and_variant_when_the_counter_is_large() -> None:
    # Start with a random part that is all ones in the 12 bits before the variant bits.
    with (
        patch("coola.identifier.monotonic.time.time_ns", return_value=1_000 * 10**6),
        patch.object(uuid7_module._MONOTONIC, "_fresh_random", return_value=2**74 - 3),
    ):
        values = [generate_uuid7() for _ in range(5)]
    assert all(UUID7_PATTERN.match(v) for v in values)
    assert values == sorted(values)
    assert len(set(values)) == 5  # the overflow borrowed the next millisecond


def test_generate_uuid7_explicit_timestamp_does_not_use_the_monotonic_state() -> None:
    with patch("coola.identifier.monotonic.time.time_ns", return_value=5_000 * 10**6):
        current = generate_uuid7()
    old = generate_uuid7(timestamp_ms=1)
    assert extract_uuid7_timestamp_ms(old) == 1
    assert old < current
    with patch("coola.identifier.monotonic.time.time_ns", return_value=5_000 * 10**6):
        assert generate_uuid7() > current
