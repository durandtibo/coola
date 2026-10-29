from __future__ import annotations

import pytest

from coola.rounding.common import MISSING_PACKAGE_POLICIES, MODES, get_scale, validate_arguments
from coola.testing.marks import numpy_available
from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np

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


def test_validate_arguments_returns_decimals() -> None:
    assert validate_arguments("round", 3) == 3


@numpy_available
def test_validate_arguments_numpy_integer_returns_int() -> None:
    result = validate_arguments("round", np.int64(2))  # ty: ignore[invalid-argument-type]
    assert type(result) is int


@numpy_available
def test_validate_arguments_numpy_integer_negative() -> None:
    with pytest.raises(ValueError, match=r"decimals must be greater than or equal to 0"):
        validate_arguments("round", np.int64(-1))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("policy", MISSING_PACKAGE_POLICIES)
def test_validate_arguments_valid_on_missing_package(policy: str) -> None:
    assert validate_arguments("round", 0, policy) == 0


def test_validate_arguments_incorrect_on_missing_package() -> None:
    with pytest.raises(
        ValueError,
        match=r"^on_missing_package must be one of \('raise', 'warn', 'ignore'\), got 'skip'$",
    ):
        validate_arguments("round", 0, "skip")


def test_validate_arguments_on_missing_package_defaults_to_raise() -> None:
    assert validate_arguments("round", 0) == 0
