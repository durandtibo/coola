from __future__ import annotations

import pytest

from coola.identifier.validation import (
    singleton_generator,
    validate_bit_range,
    validate_non_negative,
    validate_positive,
    validate_timestamp_ms,
)

###########################################
#     Tests for validate_bit_range     #
###########################################


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


############################################
#     Tests for validate_timestamp_ms     #
############################################


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


#######################################
#     Tests for validate_positive     #
#######################################


def test_validate_positive_one_is_valid() -> None:
    validate_positive(1, name="value")


def test_validate_positive_large_value_is_valid() -> None:
    validate_positive(1_000_000, name="value")


def test_validate_positive_zero_raises() -> None:
    with pytest.raises(ValueError, match="value must be positive, got 0"):
        validate_positive(0, name="value")


def test_validate_positive_negative_raises() -> None:
    with pytest.raises(ValueError, match="value must be positive, got -1"):
        validate_positive(-1, name="value")


def test_validate_positive_uses_name_in_message() -> None:
    with pytest.raises(ValueError, match=r"^length must be positive"):
        validate_positive(0, name="length")


###########################################
#     Tests for validate_non_negative     #
###########################################


def test_validate_non_negative_zero_is_valid() -> None:
    validate_non_negative(0, name="value")


def test_validate_non_negative_positive_is_valid() -> None:
    validate_non_negative(1_000_000, name="value")


def test_validate_non_negative_negative_raises() -> None:
    with pytest.raises(ValueError, match="value must be non-negative, got -1"):
        validate_non_negative(-1, name="value")


def test_validate_non_negative_uses_name_in_message() -> None:
    with pytest.raises(ValueError, match=r"^min_length must be non-negative"):
        validate_non_negative(-1, name="min_length")


###########################################
#     Tests for singleton_generator     #
###########################################


def test_singleton_generator_returns_same_instance_across_calls() -> None:
    class Counter:
        pass

    get_instance = singleton_generator(Counter)
    assert get_instance() is get_instance()


def test_singleton_generator_is_lazy() -> None:
    created = []

    class Counter:
        def __init__(self) -> None:
            created.append(1)

    get_instance = singleton_generator(Counter)
    assert created == []
    get_instance()
    assert created == [1]
    get_instance()
    assert created == [1]


def test_singleton_generator_independent_across_calls_to_singleton_generator() -> None:
    class Counter:
        pass

    get_instance_1 = singleton_generator(Counter)
    get_instance_2 = singleton_generator(Counter)
    assert get_instance_1() is not get_instance_2()
