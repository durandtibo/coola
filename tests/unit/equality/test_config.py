from __future__ import annotations

from dataclasses import replace

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tolerance import Tolerance

####################################
#     Tests for EqualityConfig     #
####################################


def test_equality_config_default_equal_nan() -> None:
    assert EqualityConfig().equal_nan is False


def test_equality_config_default_atol() -> None:
    assert EqualityConfig().atol == 0.0


def test_equality_config_default_rtol() -> None:
    assert EqualityConfig().rtol == 0.0


def test_equality_config_with_parameters_equal_nan() -> None:
    assert EqualityConfig(equal_nan=True).equal_nan is True


def test_equality_config_with_parameters_atol() -> None:
    assert EqualityConfig(tolerance=Tolerance(atol=1e-6)).atol == 1e-6


def test_equality_config_with_parameters_rtol() -> None:
    assert EqualityConfig(tolerance=Tolerance(rtol=1e-5)).rtol == 1e-5


def test_equality_config_negative_atol() -> None:
    with pytest.raises(ValueError, match="atol must be greater than or equal to 0"):
        EqualityConfig(tolerance=Tolerance(atol=-1.0))


def test_equality_config_negative_rtol() -> None:
    with pytest.raises(ValueError, match="rtol must be greater than or equal to 0"):
        EqualityConfig(tolerance=Tolerance(rtol=-0.5))


def test_equality_config_is_frozen() -> None:
    with pytest.raises(AttributeError, match="cannot assign to field 'equal_nan'"):
        EqualityConfig().equal_nan = True  # ty: ignore[invalid-assignment]


def test_equality_config_replace_keeps_the_other_settings() -> None:
    assert replace(EqualityConfig(equal_nan=True), max_depth=5).equal_nan is True


def test_equality_config_unknown_option() -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        EqualityConfig(unknown=1)  # ty: ignore[unknown-argument]


def test_equality_config_zero_atol() -> None:
    assert EqualityConfig(tolerance=Tolerance(atol=0.0, rtol=0.0)).atol == 0.0


def test_equality_config_zero_rtol() -> None:
    assert EqualityConfig(tolerance=Tolerance(atol=0.0, rtol=0.0)).rtol == 0.0


def test_equality_config_max_depth_default() -> None:
    config = EqualityConfig()
    assert config.max_depth == 100


def test_equality_config_max_depth_custom() -> None:
    config = EqualityConfig(max_depth=100)
    assert config.max_depth == 100


def test_equality_config_max_depth_zero() -> None:
    with pytest.raises(ValueError, match="max_depth must be greater than 0"):
        EqualityConfig(max_depth=0)


def test_equality_config_max_depth_negative() -> None:
    with pytest.raises(ValueError, match="max_depth must be greater than 0"):
        EqualityConfig(max_depth=-1)


def test_equality_config_eq_true() -> None:
    assert EqualityConfig() == EqualityConfig()


def test_equality_config_eq_true_with_parameters() -> None:
    assert EqualityConfig(
        equal_nan=True, tolerance=Tolerance(atol=1e-6, rtol=1e-5)
    ) == EqualityConfig(equal_nan=True, tolerance=Tolerance(atol=1e-6, rtol=1e-5))


def test_equality_config_eq_false_different_equal_nan() -> None:
    assert EqualityConfig() != EqualityConfig(equal_nan=True)


def test_equality_config_eq_false_different_atol() -> None:
    assert EqualityConfig() != EqualityConfig(tolerance=Tolerance(atol=1e-6))


def test_equality_config_eq_false_different_rtol() -> None:
    assert EqualityConfig() != EqualityConfig(tolerance=Tolerance(rtol=1e-6))


def test_equality_config_eq_false_different_max_depth() -> None:
    assert EqualityConfig() != EqualityConfig(max_depth=50)


@pytest.mark.parametrize("value", [True, 1.5, "a", None])
def test_equality_config_max_depth_wrong_type(value: object) -> None:
    with pytest.raises(TypeError, match="max_depth must be an int"):
        EqualityConfig(max_depth=value)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("name", ["atol", "rtol"])
def test_equality_config_tolerance_int(name: str) -> None:
    assert getattr(EqualityConfig(tolerance=Tolerance(**{name: 1})), name) == 1
