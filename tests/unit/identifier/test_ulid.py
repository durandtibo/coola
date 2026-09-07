from __future__ import annotations

import re

import pytest

from coola.identifier.ulid import generate_ulid

ULID_PATTERN = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")


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
