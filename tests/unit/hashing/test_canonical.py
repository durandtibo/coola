from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from coola.hashing import HasherRegistry
from coola.hashing.canonical import DecimalHasher, FractionHasher, RangeHasher


@pytest.fixture
def registry() -> HasherRegistry:
    return HasherRegistry()


###################################
#     Tests for DecimalHasher     #
###################################


def test_decimal_hasher_repr() -> None:
    assert repr(DecimalHasher()) == "DecimalHasher()"


def test_decimal_hasher_str() -> None:
    assert str(DecimalHasher()) == "DecimalHasher()"


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (Decimal("1.0"), Decimal("1.00")),
        (Decimal(1), Decimal("1.000")),
        (Decimal(100), Decimal("1E+2")),
        (Decimal(0), Decimal("0.0")),
        (Decimal("NaN"), Decimal("NaN")),
        (Decimal("Infinity"), Decimal("Infinity")),
    ],
)
def test_decimal_hasher_equal_values(
    first: Decimal, second: Decimal, registry: HasherRegistry
) -> None:
    hasher = DecimalHasher()
    assert hasher.hash(first, registry) == hasher.hash(second, registry)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (Decimal(1), Decimal(2)),
        (Decimal("1.5"), Decimal("1.50001")),
        (Decimal("Infinity"), Decimal("-Infinity")),
        (Decimal(1), Decimal("NaN")),
    ],
)
def test_decimal_hasher_different_values(
    first: Decimal, second: Decimal, registry: HasherRegistry
) -> None:
    hasher = DecimalHasher()
    assert hasher.hash(first, registry) != hasher.hash(second, registry)


def test_decimal_hasher_signaling_nan(registry: HasherRegistry) -> None:
    hasher = DecimalHasher()
    assert hasher.hash(Decimal("sNaN"), registry) == hasher.hash(Decimal("sNaN"), registry)
    assert hasher.hash(Decimal("sNaN"), registry) != hasher.hash(Decimal("NaN"), registry)


def test_decimal_hasher_length(registry: HasherRegistry) -> None:
    assert len(DecimalHasher().hash(Decimal(1), registry, length=16)) == 16


####################################
#     Tests for FractionHasher     #
####################################


def test_fraction_hasher_repr() -> None:
    assert repr(FractionHasher()) == "FractionHasher()"


def test_fraction_hasher_equal_values(registry: HasherRegistry) -> None:
    hasher = FractionHasher()
    assert hasher.hash(Fraction(1, 2), registry) == hasher.hash(Fraction(2, 4), registry)
    assert hasher.hash(Fraction(1, 2), registry) == hasher.hash(Fraction("0.5"), registry)


def test_fraction_hasher_different_values(registry: HasherRegistry) -> None:
    hasher = FractionHasher()
    assert hasher.hash(Fraction(1, 2), registry) != hasher.hash(Fraction(1, 3), registry)
    assert hasher.hash(Fraction(1, 2), registry) != hasher.hash(Fraction(2, 1), registry)
    assert hasher.hash(Fraction(1, 2), registry) != hasher.hash(Fraction(-1, 2), registry)


def test_fraction_hasher_numerator_denominator_boundary(registry: HasherRegistry) -> None:
    hasher = FractionHasher()
    assert hasher.hash(Fraction(1, 23), registry) != hasher.hash(Fraction(12, 3), registry)


def test_fraction_hasher_length(registry: HasherRegistry) -> None:
    assert len(FractionHasher().hash(Fraction(1, 2), registry, length=32)) == 32


#################################
#     Tests for RangeHasher     #
#################################


def test_range_hasher_repr() -> None:
    assert repr(RangeHasher()) == "RangeHasher()"


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (range(0), range(5, 5)),
        (range(0), range(10, 0)),
        (range(3, 4), range(3, 100, 100)),
        (range(0, 10, 2), range(0, 9, 2)),
        (range(5), range(0, 5, 1)),
    ],
)
def test_range_hasher_equal_ranges(first: range, second: range, registry: HasherRegistry) -> None:
    assert first == second
    hasher = RangeHasher()
    assert hasher.hash(first, registry) == hasher.hash(second, registry)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (range(5), range(6)),
        (range(5), range(1, 6)),
        (range(0, 10, 2), range(0, 10, 3)),
        (range(0), range(1)),
        (range(3, 4), range(4, 5)),
        (range(0, 6, 2), range(0, 6, 3)),
    ],
)
def test_range_hasher_different_ranges(
    first: range, second: range, registry: HasherRegistry
) -> None:
    assert first != second
    hasher = RangeHasher()
    assert hasher.hash(first, registry) != hasher.hash(second, registry)


def test_range_hasher_huge_range_is_fast(registry: HasherRegistry) -> None:
    hasher = RangeHasher()
    assert hasher.hash(range(10**18), registry) != hasher.hash(range(10**18 + 1), registry)


def test_range_hasher_length(registry: HasherRegistry) -> None:
    assert len(RangeHasher().hash(range(3), registry, length=8)) == 8


def test_decimal_hasher_hash_beyond_context_precision(registry: HasherRegistry) -> None:
    hasher = DecimalHasher()
    first = Decimal("1." + "0" * 30 + "1")
    second = Decimal("1." + "0" * 30 + "2")
    assert hasher.hash(first, registry) != hasher.hash(second, registry)


def test_decimal_hasher_hash_ignores_context(registry: HasherRegistry) -> None:
    hasher = DecimalHasher()
    value = Decimal("1.234567890123456789")
    expected = hasher.hash(value, registry)
    with localcontext() as ctx:
        ctx.prec = 3
        assert hasher.hash(value, registry) == expected


def test_decimal_hasher_hash_negative_zero(registry: HasherRegistry) -> None:
    hasher = DecimalHasher()
    assert hasher.hash(Decimal("-0"), registry) == hasher.hash(Decimal("0.00"), registry)
