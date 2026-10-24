from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from fractions import Fraction

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import DifferenceCloseEqualityTester


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


def test_difference_close_equality_tester_str() -> None:
    assert str(DifferenceCloseEqualityTester()).startswith("DifferenceCloseEqualityTester(")


def test_difference_close_equality_tester_equal(config: EqualityConfig) -> None:
    assert DifferenceCloseEqualityTester().objects_are_equal(Decimal(1), Decimal(1), config)


def test_difference_close_equality_tester_same_object(config: EqualityConfig) -> None:
    value = Fraction(1, 3)
    assert DifferenceCloseEqualityTester().objects_are_equal(value, value, config)


def test_difference_close_equality_tester_different(config: EqualityConfig) -> None:
    assert not DifferenceCloseEqualityTester().objects_are_equal(
        timedelta(seconds=1), timedelta(seconds=2), config
    )


def test_difference_close_equality_tester_different_type(config: EqualityConfig) -> None:
    assert not DifferenceCloseEqualityTester().objects_are_equal(Decimal(1), Fraction(1), config)


def test_difference_close_equality_tester_atol() -> None:
    assert DifferenceCloseEqualityTester().objects_are_equal(
        Decimal("1.0"), Decimal("1.00001"), EqualityConfig(atol=1e-3)
    )
