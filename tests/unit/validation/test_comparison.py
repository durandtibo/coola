from __future__ import annotations

from datetime import date

import pytest

from coola.validation import (
    validate_closed_range,
    validate_ge,
    validate_gt,
    validate_half_open_range,
    validate_le,
    validate_lt,
    validate_negative,
    validate_non_negative,
    validate_non_positive,
    validate_open_range,
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


@pytest.mark.parametrize(
    "validate",
    [
        pytest.param(lambda v: validate_ge(v, 0), id="ge"),
        pytest.param(lambda v: validate_gt(v, 0), id="gt"),
        pytest.param(lambda v: validate_le(v, 0), id="le"),
        pytest.param(lambda v: validate_lt(v, 0), id="lt"),
        pytest.param(validate_negative, id="negative"),
        pytest.param(validate_non_negative, id="non_negative"),
        pytest.param(validate_non_positive, id="non_positive"),
        pytest.param(validate_positive, id="positive"),
    ],
)
def test_validate_comparison_rejects_nan(validate: object) -> None:
    with pytest.raises(ValueError, match="nan"):
        validate(float("nan"))  # type: ignore[operator]


@pytest.mark.parametrize("value", [None, "a"])
def test_validate_non_negative_not_a_number_raises(value: object) -> None:
    with pytest.raises(TypeError, match="value cannot be compared with 0, got"):
        validate_non_negative(value)


def test_validate_non_negative_list_raises() -> None:
    with pytest.raises(TypeError, match="value cannot be compared with 0, got"):
        validate_non_negative([1, 2])


def test_validate_non_negative_nan_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        validate_non_negative(float("nan"))


def test_validate_open_range_inside() -> None:
    validate_open_range(0.5, 0, 1)


def test_validate_open_range_low_invalid() -> None:
    with pytest.raises(ValueError, match=r"value must be in the open range \(0, 1\), got 0"):
        validate_open_range(0, 0, 1)


def test_validate_open_range_high_invalid() -> None:
    with pytest.raises(ValueError, match=r"value must be in the open range \(0, 1\), got 1"):
        validate_open_range(1, 0, 1)


def test_validate_open_range_custom_name() -> None:
    with pytest.raises(ValueError, match="count must be in the"):
        validate_open_range(5, 0, 1, name="count")


def test_validate_open_range_nan() -> None:
    with pytest.raises(ValueError, match="value must be in the"):
        validate_open_range(float("nan"), 0, 1)


def test_validate_open_range_none() -> None:
    with pytest.raises(TypeError, match="value cannot be compared with 0 and 1, got None"):
        validate_open_range(None, 0, 1)


def test_validate_closed_range_inside() -> None:
    validate_closed_range(0.5, 0, 1)


def test_validate_closed_range_low() -> None:
    validate_closed_range(0, 0, 1)


def test_validate_closed_range_high() -> None:
    validate_closed_range(1, 0, 1)


def test_validate_closed_range_below_invalid() -> None:
    with pytest.raises(ValueError, match=r"value must be in the closed range \[0, 1\], got -1"):
        validate_closed_range(-1, 0, 1)


def test_validate_closed_range_above_invalid() -> None:
    with pytest.raises(ValueError, match=r"value must be in the closed range \[0, 1\], got 2"):
        validate_closed_range(2, 0, 1)


def test_validate_closed_range_custom_name() -> None:
    with pytest.raises(ValueError, match="count must be in the"):
        validate_closed_range(5, 0, 1, name="count")


def test_validate_closed_range_nan() -> None:
    with pytest.raises(ValueError, match="value must be in the"):
        validate_closed_range(float("nan"), 0, 1)


def test_validate_closed_range_none() -> None:
    with pytest.raises(TypeError, match="value cannot be compared with 0 and 1, got None"):
        validate_closed_range(None, 0, 1)


def test_validate_half_open_range_inside() -> None:
    validate_half_open_range(0.5, 0, 1)


def test_validate_half_open_range_low() -> None:
    validate_half_open_range(0, 0, 1)


def test_validate_half_open_range_below_invalid() -> None:
    with pytest.raises(ValueError, match=r"value must be in the half-open range \[0, 1\), got -1"):
        validate_half_open_range(-1, 0, 1)


def test_validate_half_open_range_high_invalid() -> None:
    with pytest.raises(ValueError, match=r"value must be in the half-open range \[0, 1\), got 1"):
        validate_half_open_range(1, 0, 1)


def test_validate_half_open_range_custom_name() -> None:
    with pytest.raises(ValueError, match="count must be in the"):
        validate_half_open_range(5, 0, 1, name="count")


def test_validate_half_open_range_nan() -> None:
    with pytest.raises(ValueError, match="value must be in the"):
        validate_half_open_range(float("nan"), 0, 1)


def test_validate_half_open_range_none() -> None:
    with pytest.raises(TypeError, match="value cannot be compared with 0 and 1, got None"):
        validate_half_open_range(None, 0, 1)


def test_validate_ge_message_shows_the_value_with_repr() -> None:
    with pytest.raises(ValueError, match=r"^value must be greater than or equal to 'b', got 'a'$"):
        validate_ge("a", "b")


def test_validate_ge_incomparable_message() -> None:
    with pytest.raises(TypeError, match=r"^value cannot be compared with 0, got None$"):
        validate_ge(None, 0)


def test_validate_ge_incomparable_keeps_the_cause() -> None:
    with pytest.raises(TypeError) as exc_info:
        validate_ge(None, 0)
    assert isinstance(exc_info.value.__cause__, TypeError)


def test_validate_ge_accepts_dates() -> None:
    validate_ge(date(2024, 1, 1), date(2020, 1, 1))


def test_validate_ge_nan_raises() -> None:
    with pytest.raises(ValueError, match=r"^value must be greater than or equal to 0, got nan$"):
        validate_ge(float("nan"), 0)


def test_validate_closed_range_message_shows_the_bounds_with_repr() -> None:
    with pytest.raises(
        ValueError, match=r"^value must be in the closed range \['a', 'c'\], got 'd'$"
    ):
        validate_closed_range("d", "a", "c")
