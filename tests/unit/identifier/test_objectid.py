from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from coola.identifier import (
    extract_object_id_timestamp_s,
    generate_object_id,
    objectid as objectid_module,
)

if TYPE_CHECKING:
    from collections.abc import Generator

OBJECT_ID_PATTERN = re.compile(r"^[0-9a-f]{24}$")
TIME_NS = "coola.identifier.sequence.time.time_ns"


@pytest.fixture(autouse=True)
def _reset_sequence_state() -> Generator[None]:
    """Reset the state of the sequence, which is shared by the tests of
    the process and depends on the time (a test with a fake clock must
    start from scratch)."""
    objectid_module._SEQUENCE._reset_after_fork()
    yield
    objectid_module._SEQUENCE._reset_after_fork()


########################################
#     Tests for generate_object_id     #
########################################


def test_generate_object_id_returns_str() -> None:
    assert isinstance(generate_object_id(), str)


def test_generate_object_id_length() -> None:
    assert len(generate_object_id()) == 24


def test_generate_object_id_matches_pattern() -> None:
    assert OBJECT_ID_PATTERN.match(generate_object_id())


def test_generate_object_id_is_unique() -> None:
    assert generate_object_id() != generate_object_id()


def test_generate_object_id_encodes_the_current_time() -> None:
    with patch(TIME_NS, return_value=1_700_000_000 * 10**9):
        object_id = generate_object_id()
    assert object_id[:8] == "6553f100"


def test_generate_object_id_increases_in_the_same_second() -> None:
    with patch(TIME_NS, return_value=1_700_000_000 * 10**9):
        first = generate_object_id()
        second = generate_object_id()
    assert first < second


def test_generate_object_id_increments_the_random_part_in_the_same_second() -> None:
    with patch(TIME_NS, return_value=1_700_000_000 * 10**9):
        first = generate_object_id()
        second = generate_object_id()
    assert int(second, 16) == int(first, 16) + 1


def test_generate_object_id_increases_in_a_new_second() -> None:
    with patch(TIME_NS, return_value=1_700_000_000 * 10**9):
        first = generate_object_id()
    with patch(TIME_NS, return_value=1_700_000_001 * 10**9):
        second = generate_object_id()
    assert first < second


def test_generate_object_id_keeps_increasing_if_the_clock_moves_backward() -> None:
    with patch(TIME_NS, return_value=1_700_000_000 * 10**9):
        first = generate_object_id()
    with patch(TIME_NS, return_value=1_699_999_990 * 10**9):
        second = generate_object_id()
    assert int(second, 16) == int(first, 16) + 1


def test_generate_object_id_borrows_the_next_second_when_the_counter_is_exhausted() -> None:
    with (
        patch(TIME_NS, return_value=1_700_000_000 * 10**9),
        patch.object(objectid_module._SEQUENCE, "_start", return_value=2**64 - 1),
    ):
        generate_object_id()
        object_id = generate_object_id()
    assert object_id[:8] == "6553f101"


def test_generate_object_id_explicit_timestamp() -> None:
    assert generate_object_id(timestamp_s=1_700_000_000)[:8] == "6553f100"


def test_generate_object_id_explicit_timestamp_zero() -> None:
    assert generate_object_id(timestamp_s=0)[:8] == "00000000"


def test_generate_object_id_explicit_timestamp_maximum() -> None:
    assert generate_object_id(timestamp_s=2**32 - 1)[:8] == "ffffffff"


def test_generate_object_id_explicit_timestamp_random_part_is_random() -> None:
    assert generate_object_id(timestamp_s=42) != generate_object_id(timestamp_s=42)


def test_generate_object_id_explicit_timestamp_does_not_use_the_sequence() -> None:
    with patch(TIME_NS, return_value=1_700_000_000 * 10**9):
        first = generate_object_id()
    generate_object_id(timestamp_s=1)
    with patch(TIME_NS, return_value=1_700_000_000 * 10**9):
        second = generate_object_id()
    assert int(second, 16) == int(first, 16) + 1


def test_generate_object_id_negative_timestamp_raises() -> None:
    with pytest.raises(ValueError, match="timestamp_s must be in the range"):
        generate_object_id(timestamp_s=-1)


def test_generate_object_id_timestamp_too_large_raises() -> None:
    with pytest.raises(ValueError, match="timestamp_s must be in the range"):
        generate_object_id(timestamp_s=2**32)


@pytest.mark.parametrize("timestamp_s", [True, 1.5, "1"])
def test_generate_object_id_timestamp_wrong_type_raises(timestamp_s: object) -> None:
    with pytest.raises(TypeError, match="timestamp_s must be an int"):
        generate_object_id(timestamp_s=timestamp_s)  # ty: ignore[invalid-argument-type]


def test_generate_object_id_timestamp_is_keyword_only() -> None:
    with pytest.raises(TypeError, match="positional argument"):
        generate_object_id(1_700_000_000)  # ty: ignore[too-many-positional-arguments]


def test_generate_object_id_current_time_beyond_32_bits_raises() -> None:
    with (
        patch(TIME_NS, return_value=2**32 * 10**9),
        pytest.raises(ValueError, match="timestamp_s must be in the range"),
    ):
        generate_object_id()


###################################################
#     Tests for extract_object_id_timestamp_s     #
###################################################


def test_extract_object_id_timestamp_s_roundtrip() -> None:
    assert extract_object_id_timestamp_s(generate_object_id(timestamp_s=1_800_000_000)) == (
        1_800_000_000
    )


def test_extract_object_id_timestamp_s_returns_int() -> None:
    assert isinstance(extract_object_id_timestamp_s(generate_object_id()), int)


def test_extract_object_id_timestamp_s_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="24-character lowercase hex string"):
        extract_object_id_timestamp_s("too-short")


def test_extract_object_id_timestamp_s_invalid_hex_raises() -> None:
    with pytest.raises(ValueError, match="24-character lowercase hex string"):
        extract_object_id_timestamp_s("z" * 24)


###################################################
#     Tests for the strict inputs and inverse     #
###################################################


@pytest.mark.parametrize("object_id", [None, 123, b"65920a80aabbccddeeff0011"])
def test_extract_object_id_timestamp_s_not_a_str_raises(object_id: object) -> None:
    with pytest.raises(TypeError, match="object_id must be an instance of"):
        extract_object_id_timestamp_s(object_id)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "object_id",
    [
        "65 92 0a 80 aa bb cc dd ee ff 00 11"[:24],
        "65920A80AABBCCDDEEFF0011",
        "65920a80aabbccddeeff0011\n",
        " 65920a80aabbccddeeff001",
    ],
)
def test_extract_object_id_timestamp_s_not_lowercase_hex_raises(object_id: str) -> None:
    with pytest.raises(ValueError, match="24-character lowercase hex string"):
        extract_object_id_timestamp_s(object_id)
