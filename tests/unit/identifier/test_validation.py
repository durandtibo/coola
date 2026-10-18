from __future__ import annotations

import time

import pytest

from coola.identifier.validation import (
    resolve_timestamp_ms,
    validate_bit_range,
    validate_timestamp_ms,
)

########################################
#     Tests for validate_bit_range     #
########################################


def test_validate_bit_range_minimum_is_valid() -> None:
    validate_bit_range(0, 8, name="value")


def test_validate_bit_range_maximum_is_valid() -> None:
    validate_bit_range(255, 8, name="value")


def test_validate_bit_range_mid_range_is_valid() -> None:
    validate_bit_range(128, 8, name="value")


def test_validate_bit_range_negative_raises() -> None:
    with pytest.raises(ValueError, match="value must fit in 8 bits \\(0 to 255\\), got -1"):
        validate_bit_range(-1, 8, name="value")


def test_validate_bit_range_too_large_raises() -> None:
    with pytest.raises(ValueError, match="value must fit in 8 bits \\(0 to 255\\), got 256"):
        validate_bit_range(256, 8, name="value")


def test_validate_bit_range_uses_name_in_message() -> None:
    with pytest.raises(ValueError, match=r"^worker_id must fit in 10 bits"):
        validate_bit_range(-1, 10, name="worker_id")


###########################################
#     Tests for validate_timestamp_ms     #
###########################################


def test_validate_timestamp_ms_zero_is_valid() -> None:
    validate_timestamp_ms(0)


def test_validate_timestamp_ms_max_is_valid() -> None:
    validate_timestamp_ms(2**48 - 1)


def test_validate_timestamp_ms_typical_value_is_valid() -> None:
    validate_timestamp_ms(1_800_000_000_000)


def test_validate_timestamp_ms_negative_raises() -> None:
    with pytest.raises(ValueError, match="timestamp_ms must fit in 48 bits"):
        validate_timestamp_ms(-1)


def test_validate_timestamp_ms_too_large_raises() -> None:
    with pytest.raises(ValueError, match="timestamp_ms must fit in 48 bits"):
        validate_timestamp_ms(2**48)


##########################################
#     Tests for resolve_timestamp_ms     #
##########################################


def test_resolve_timestamp_ms_returns_the_given_timestamp() -> None:
    assert resolve_timestamp_ms(1_234_567) == 1_234_567


def test_resolve_timestamp_ms_none_returns_the_current_time() -> None:
    before = time.time_ns() // 1_000_000
    resolved = resolve_timestamp_ms(None)
    after = time.time_ns() // 1_000_000
    assert before <= resolved <= after


@pytest.mark.parametrize("timestamp_ms", [-1, 2**48])
def test_resolve_timestamp_ms_invalid_timestamp_raises(timestamp_ms: int) -> None:
    with pytest.raises(ValueError, match="timestamp_ms must fit in 48 bits"):
        resolve_timestamp_ms(timestamp_ms)
