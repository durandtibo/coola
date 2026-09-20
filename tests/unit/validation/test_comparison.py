from __future__ import annotations

import pytest

from coola.validation import (
    validate_ge,
    validate_gt,
    validate_le,
    validate_lt,
    validate_negative,
    validate_non_negative,
    validate_non_positive,
    validate_positive,
)


def test_validate_ge_greater() -> None:
    validate_ge(1, 0)


def test_validate_ge_equal() -> None:
    validate_ge(1, 1)


def test_validate_ge_lower() -> None:
    with pytest.raises(ValueError, match="value must be greater than or equal to 1, got 0"):
        validate_ge(0, 1)


def test_validate_ge_custom_name() -> None:
    with pytest.raises(ValueError, match="count must be greater than or equal to 0, got -1"):
        validate_ge(-1, 0, name="count")


def test_validate_gt_greater() -> None:
    validate_gt(1, 0)


def test_validate_gt_equal() -> None:
    with pytest.raises(ValueError, match="value must be greater than 1, got 1"):
        validate_gt(1, 1)


def test_validate_gt_lower() -> None:
    with pytest.raises(ValueError, match="value must be greater than 1, got 0"):
        validate_gt(0, 1)


def test_validate_gt_custom_name() -> None:
    with pytest.raises(ValueError, match="count must be greater than 0, got -1"):
        validate_gt(-1, 0, name="count")


def test_validate_le_lower() -> None:
    validate_le(0, 1)


def test_validate_le_equal() -> None:
    validate_le(1, 1)


def test_validate_le_greater() -> None:
    with pytest.raises(ValueError, match="value must be less than or equal to 0, got 1"):
        validate_le(1, 0)


def test_validate_le_custom_name() -> None:
    with pytest.raises(ValueError, match="count must be less than or equal to 0, got 1"):
        validate_le(1, 0, name="count")


def test_validate_lt_lower() -> None:
    validate_lt(0, 1)


def test_validate_lt_equal() -> None:
    with pytest.raises(ValueError, match="value must be less than 1, got 1"):
        validate_lt(1, 1)


def test_validate_lt_greater() -> None:
    with pytest.raises(ValueError, match="value must be less than 0, got 1"):
        validate_lt(1, 0)


def test_validate_lt_custom_name() -> None:
    with pytest.raises(ValueError, match="count must be less than 0, got 1"):
        validate_lt(1, 0, name="count")


def test_validate_non_negative_zero() -> None:
    validate_non_negative(0)


def test_validate_non_negative_positive() -> None:
    validate_non_negative(1)


def test_validate_non_negative_negative() -> None:
    with pytest.raises(ValueError, match="value must be non-negative, got -1"):
        validate_non_negative(-1)


def test_validate_non_negative_custom_name() -> None:
    with pytest.raises(ValueError, match="count must be non-negative, got -1"):
        validate_non_negative(-1, name="count")


def test_validate_positive_positive() -> None:
    validate_positive(1)


def test_validate_positive_zero() -> None:
    with pytest.raises(ValueError, match="value must be positive, got 0"):
        validate_positive(0)


def test_validate_positive_negative() -> None:
    with pytest.raises(ValueError, match="value must be positive, got -1"):
        validate_positive(-1)


def test_validate_positive_custom_name() -> None:
    with pytest.raises(ValueError, match="count must be positive, got 0"):
        validate_positive(0, name="count")


def test_validate_negative_negative() -> None:
    validate_negative(-1)


def test_validate_negative_zero() -> None:
    with pytest.raises(ValueError, match="value must be negative, got 0"):
        validate_negative(0)


def test_validate_negative_positive() -> None:
    with pytest.raises(ValueError, match="value must be negative, got 1"):
        validate_negative(1)


def test_validate_negative_custom_name() -> None:
    with pytest.raises(ValueError, match="count must be negative, got 0"):
        validate_negative(0, name="count")


def test_validate_non_positive_zero() -> None:
    validate_non_positive(0)


def test_validate_non_positive_negative() -> None:
    validate_non_positive(-1)


def test_validate_non_positive_positive() -> None:
    with pytest.raises(ValueError, match="value must be non-positive, got 1"):
        validate_non_positive(1)


def test_validate_non_positive_custom_name() -> None:
    with pytest.raises(ValueError, match="count must be non-positive, got 1"):
        validate_non_positive(1, name="count")
