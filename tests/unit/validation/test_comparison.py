from __future__ import annotations

import pytest

from coola.validation import validate_ge, validate_gt, validate_le, validate_lt


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
