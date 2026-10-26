from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from fractions import Fraction

import pytest

from coola.equality._helpers.difference import difference_close
from coola.equality.config import EqualityConfig
from coola.equality.tolerance import Tolerance


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


###################################
#     Tests for difference_close  #
###################################


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(Decimal("1.0"), Decimal("1.00001"), id="decimal"),
        pytest.param(Fraction(1, 3), Fraction(333333, 1000000), id="fraction"),
        pytest.param(timedelta(seconds=1), timedelta(seconds=1.0001), id="timedelta"),
    ],
)
def test_difference_close_atol(actual: object, expected: object) -> None:
    assert difference_close(actual, expected, EqualityConfig(tolerance=Tolerance(atol=1e-3)))


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(Decimal("1.0"), Decimal("1.00001"), id="decimal"),
        pytest.param(Fraction(1, 3), Fraction(333333, 1000000), id="fraction"),
        pytest.param(timedelta(seconds=1), timedelta(seconds=1.0001), id="timedelta"),
    ],
)
def test_difference_close_without_tolerance(
    actual: object, expected: object, config: EqualityConfig
) -> None:
    assert not difference_close(actual, expected, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(Decimal("1.0"), Decimal("1.0"), id="decimal"),
        pytest.param(Fraction(1, 3), Fraction(1, 3), id="fraction"),
        pytest.param(timedelta(seconds=1), timedelta(seconds=1), id="timedelta"),
    ],
)
def test_difference_close_equal_without_tolerance(
    actual: object, expected: object, config: EqualityConfig
) -> None:
    assert difference_close(actual, expected, config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(Decimal("1.0"), Decimal("1.1"), id="decimal"),
        pytest.param(Fraction(1, 3), Fraction(1, 2), id="fraction"),
        pytest.param(timedelta(seconds=1), timedelta(seconds=2), id="timedelta"),
    ],
)
def test_difference_close_outside_the_tolerance(actual: object, expected: object) -> None:
    assert not difference_close(actual, expected, EqualityConfig(tolerance=Tolerance(atol=1e-3)))


def test_difference_close_rtol_decimal() -> None:
    assert difference_close(
        Decimal(100), Decimal(101), EqualityConfig(tolerance=Tolerance(rtol=0.02))
    )


def test_difference_close_rtol_fraction() -> None:
    assert difference_close(
        Fraction(100), Fraction(101), EqualityConfig(tolerance=Tolerance(rtol=0.02))
    )


def test_difference_close_rtol_timedelta() -> None:
    assert difference_close(
        timedelta(seconds=100),
        timedelta(seconds=101),
        EqualityConfig(tolerance=Tolerance(rtol=0.02)),
    )


def test_difference_close_is_symmetric_with_rtol() -> None:
    config = EqualityConfig(tolerance=Tolerance(rtol=0.01))
    assert difference_close(Decimal("101.005"), Decimal(100), config) == difference_close(
        Decimal(100), Decimal("101.005"), config
    )


def test_difference_close_decimal_nan_without_equal_nan(config: EqualityConfig) -> None:
    assert not difference_close(Decimal("NaN"), Decimal("NaN"), config)


def test_difference_close_decimal_nan_with_equal_nan() -> None:
    assert difference_close(Decimal("NaN"), Decimal("NaN"), EqualityConfig(equal_nan=True))


def test_difference_close_decimal_nan_and_number_with_equal_nan() -> None:
    assert not difference_close(Decimal("NaN"), Decimal(1), EqualityConfig(equal_nan=True))


def test_difference_close_decimal_nan_with_tolerance() -> None:
    assert not difference_close(
        Decimal("NaN"), Decimal("NaN"), EqualityConfig(tolerance=Tolerance(atol=1.0))
    )


def test_difference_close_decimal_infinity_equal() -> None:
    assert difference_close(
        Decimal("Infinity"), Decimal("Infinity"), EqualityConfig(tolerance=Tolerance(atol=1.0))
    )


def test_difference_close_decimal_infinity_and_number() -> None:
    assert not difference_close(
        Decimal("Infinity"), Decimal(1), EqualityConfig(tolerance=Tolerance(atol=1.0))
    )


def test_difference_close_decimal_opposite_infinities() -> None:
    assert not difference_close(
        Decimal("Infinity"), Decimal("-Infinity"), EqualityConfig(tolerance=Tolerance(atol=1.0))
    )


########################################
#     Tests for the tolerance rule     #
########################################


# |1 - 3| = 2 <= 0.5 + 0.5 * 3: the tolerances add up, as in NumPy.
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (Decimal(1), Decimal(3)),
        (Decimal(3), Decimal(1)),
        (Fraction(1), Fraction(3)),
        (Fraction(3), Fraction(1)),
        (timedelta(seconds=1), timedelta(seconds=3)),
        (timedelta(seconds=3), timedelta(seconds=1)),
    ],
)
def test_difference_close_atol_and_rtol_add_up(actual: object, expected: object) -> None:
    assert difference_close(
        actual, expected, EqualityConfig(tolerance=Tolerance(atol=0.5, rtol=0.5))
    )


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (Decimal(1), Decimal("3.5")),
        (Fraction(1), Fraction(7, 2)),
        (timedelta(seconds=1), timedelta(seconds=3.5)),
    ],
)
def test_difference_close_outside_atol_and_rtol(actual: object, expected: object) -> None:
    assert not difference_close(
        actual, expected, EqualityConfig(tolerance=Tolerance(atol=0.5, rtol=0.5))
    )
