from __future__ import annotations

import pytest

from coola.rounding.common import MODES, get_scale, validate_arguments

#################################
#     Tests for get_scale       #
#################################


def test_get_scale_zero() -> None:
    assert get_scale(0) == 1.0


def test_get_scale_two() -> None:
    assert get_scale(2) == 100.0


def test_get_scale_too_large() -> None:
    assert get_scale(400) is None


########################################
#     Tests for validate_arguments     #
########################################


@pytest.mark.parametrize("mode", MODES)
def test_validate_arguments_valid_mode(mode: str) -> None:
    validate_arguments(mode, 0)


def test_validate_arguments_incorrect_mode() -> None:
    with pytest.raises(
        ValueError, match=r"^mode must be one of \('round', 'floor', 'ceil'\), got 'trunc'$"
    ):
        validate_arguments("trunc", 0)


def test_validate_arguments_negative_decimals() -> None:
    with pytest.raises(ValueError, match=r"decimals must be greater than or equal to 0"):
        validate_arguments("round", -1)


@pytest.mark.parametrize("decimals", [1.0, "1", None, True])
def test_validate_arguments_decimals_not_int(decimals: object) -> None:
    with pytest.raises(TypeError, match=r"decimals must be an int"):
        validate_arguments("round", decimals)  # ty: ignore[invalid-argument-type]
