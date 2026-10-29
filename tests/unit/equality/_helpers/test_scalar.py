from __future__ import annotations

import enum
import math
import numbers

import pytest

from coola.equality._helpers.scalar import complex_are_equal, int_isclose, number_equal
from coola.equality.config import EqualityConfig
from coola.equality.tolerance import Tolerance


class Unconvertible:
    """A ``numbers.Real`` that cannot be converted to a ``float``."""

    def __init__(self, value: int = 0) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Unconvertible) and self.value == other.value

    __hash__ = None

    def __float__(self) -> float:
        msg = "no float"
        raise TypeError(msg)


numbers.Real.register(Unconvertible)


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


#################################
#     Tests for int_isclose     #
#################################


@pytest.mark.parametrize(
    ("number1", "number2", "rtol", "atol", "expected"),
    [
        pytest.param(10**400, 10**400, 0.0, 0.0, True, id="equal-no-tolerance"),
        pytest.param(10**400, 10**400 + 1, 0.0, 0.0, False, id="different-no-tolerance"),
        pytest.param(10**400, 10**400 + 1, 1e-5, 0.0, True, id="rtol-close"),
        pytest.param(10**400, 2 * 10**400, 1e-5, 0.0, False, id="rtol-far"),
        pytest.param(10**400, 10**400 + 1, 0.0, 1.0, True, id="atol-boundary"),
        pytest.param(10**400, 10**400 + 2, 0.0, 1.0, False, id="atol-exceeded"),
        pytest.param(10**400, 10**400 + 5, 0.0, 10.0, True, id="atol-close"),
        pytest.param(-(10**400), -(10**400) - 1, 1e-5, 0.0, True, id="negative"),
        pytest.param(-(10**400), 10**400, 1e-5, 0.0, False, id="opposite-signs"),
        pytest.param(0, 0, 0.0, 0.0, True, id="zero"),
        pytest.param(1, 2, 0.5, 0.0, True, id="small-ints-rtol"),
        pytest.param(1.5, 1.5, 0.0, 0.0, True, id="floats"),
        pytest.param(10**400, float("inf"), 1e-5, 0.0, False, id="int-inf"),
        pytest.param(float("inf"), float("inf"), 1e-5, 0.0, True, id="inf-inf"),
        pytest.param(float("inf"), float("-inf"), 1e-5, 0.0, False, id="inf-neg-inf"),
        pytest.param(float("nan"), float("nan"), 1e-5, 0.0, False, id="nan-nan"),
    ],
)
def test_int_isclose(
    number1: float, number2: float, rtol: float, atol: float, expected: bool
) -> None:
    assert int_isclose(number1, number2, rtol=rtol, atol=atol) == expected


def test_int_isclose_is_symmetric() -> None:
    assert int_isclose(10**400, 10**400 + 3, rtol=1e-5) == int_isclose(
        10**400 + 3, 10**400, rtol=1e-5
    )


def test_int_isclose_default_tolerances_are_exact() -> None:
    assert int_isclose(10**400, 10**400)


def test_int_isclose_default_tolerances_are_exact_different() -> None:
    assert not int_isclose(10**400, 10**400 + 1)


@pytest.mark.parametrize(("number1", "number2"), [(100, 101), (1000, 1001), (7, 8), (0, 1)])
def test_int_isclose_agrees_with_math_isclose_for_small_ints(number1: int, number2: int) -> None:
    assert int_isclose(number1, number2, rtol=0.01, atol=0.5) == math.isclose(
        number1, number2, rel_tol=0.01, abs_tol=0.5
    )


def test_number_equal_complex_exact() -> None:
    assert number_equal(1 + 1j, 1 + 1j, EqualityConfig())  # ty: ignore[invalid-argument-type]


def test_number_equal_complex_exact_different() -> None:
    assert not number_equal(1 + 1j, 1 + 2j, EqualityConfig())  # ty: ignore[invalid-argument-type]


def test_number_equal_complex_atol() -> None:
    assert number_equal(1 + 1j, 1 + 1.05j, EqualityConfig(tolerance=Tolerance(atol=0.1)))  # ty: ignore[invalid-argument-type]


def test_number_equal_complex_rtol() -> None:
    assert number_equal(1 + 1j, 1.05 + 1j, EqualityConfig(tolerance=Tolerance(rtol=0.1)))  # ty: ignore[invalid-argument-type]


def test_number_equal_complex_tolerance_not_equal() -> None:
    assert not number_equal(1 + 1j, 1 + 2j, EqualityConfig(tolerance=Tolerance(atol=0.1)))  # ty: ignore[invalid-argument-type]


def test_number_equal_complex_tolerance_checks_imaginary_part() -> None:
    assert not number_equal(1 + 0j, 1 + 5j, EqualityConfig(tolerance=Tolerance(atol=0.1)))  # ty: ignore[invalid-argument-type]


def test_number_equal_complex_and_real_tolerance() -> None:
    assert number_equal(1 + 0j, 1.05, EqualityConfig(tolerance=Tolerance(atol=0.1)))  # ty: ignore[invalid-argument-type]


def test_number_equal_real_and_complex_tolerance() -> None:
    assert number_equal(1.05, 1 + 0j, EqualityConfig(tolerance=Tolerance(atol=0.1)))  # ty: ignore[invalid-argument-type]


def test_number_equal_complex_with_imaginary_part_and_real_tolerance() -> None:
    assert not number_equal(1 + 1j, 1.0, EqualityConfig(tolerance=Tolerance(atol=0.1)))  # ty: ignore[invalid-argument-type]


NAN = float("nan")


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (1 + 2j, 1 + 2j),
        (complex(NAN, 1), complex(NAN, 1)),
        (complex(1, NAN), complex(1, NAN)),
        (complex(NAN, NAN), complex(NAN, NAN)),
        (complex(float("inf"), 1), complex(float("inf"), 1)),
    ],
)
def test_complex_are_equal_true(actual: complex, expected: complex) -> None:
    assert complex_are_equal(actual, expected, EqualityConfig(equal_nan=True))


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (1 + 2j, 1 + 3j),
        (complex(NAN, 1), complex(NAN, 2)),
        (complex(1, NAN), complex(2, NAN)),
        (complex(NAN, 1), complex(1, 1)),
        (complex(NAN, NAN), complex(NAN, 1)),
    ],
)
def test_complex_are_equal_false(actual: complex, expected: complex) -> None:
    assert not complex_are_equal(actual, expected, EqualityConfig(equal_nan=True))


def test_complex_are_equal_tolerance() -> None:
    config = EqualityConfig(equal_nan=True, tolerance=Tolerance(atol=0.1))
    assert complex_are_equal(complex(NAN, 1), complex(NAN, 1.05), config)


def test_complex_are_equal_tolerance_different() -> None:
    config = EqualityConfig(equal_nan=True, tolerance=Tolerance(atol=0.1))
    assert not complex_are_equal(complex(NAN, 1), complex(NAN, 1.5), config)


########################################
#     Tests for the tolerance rule     #
########################################


# |1 - 3| = 2 <= 0.5 + 0.5 * 3: the tolerances add up, as in NumPy.
@pytest.mark.parametrize(
    ("number1", "number2"),
    [(1.0, 3.0), (3.0, 1.0), (1, 3), (3, 1), (complex(1, 0), complex(3, 0))],
)
def test_number_equal_atol_and_rtol_add_up(number1: float, number2: float) -> None:
    assert number_equal(number1, number2, EqualityConfig(tolerance=Tolerance(atol=0.5, rtol=0.5)))


def test_number_equal_atol_and_rtol_add_up_huge_int() -> None:
    big = 10**400
    assert number_equal(big, 3 * big, EqualityConfig(tolerance=Tolerance(atol=0.5, rtol=1.0)))


@pytest.mark.parametrize(
    ("number1", "number2"), [(1.0, 3.5), (3.5, 1.0), (1, 4), (4, 1), (1.0, float("inf"))]
)
def test_number_equal_outside_atol_and_rtol(number1: float, number2: float) -> None:
    assert not number_equal(
        number1, number2, EqualityConfig(tolerance=Tolerance(atol=0.5, rtol=0.5))
    )


def test_number_equal_not_a_number_equal() -> None:
    assert number_equal("a", "a", EqualityConfig(tolerance=Tolerance(atol=0.5)))  # ty: ignore[invalid-argument-type]


def test_number_equal_not_a_number_different() -> None:
    assert not number_equal("a", "b", EqualityConfig(tolerance=Tolerance(atol=0.5)))  # ty: ignore[invalid-argument-type]


def test_number_equal_real_without_float_equal() -> None:
    assert number_equal(
        Unconvertible(),  # ty: ignore[invalid-argument-type]
        Unconvertible(),  # ty: ignore[invalid-argument-type]
        EqualityConfig(tolerance=Tolerance(atol=0.5)),
    )


def test_number_equal_real_without_float_different() -> None:
    assert not number_equal(
        Unconvertible(),  # ty: ignore[invalid-argument-type]
        Unconvertible(1),  # ty: ignore[invalid-argument-type]
        EqualityConfig(tolerance=Tolerance(atol=0.5)),
    )


class Level(enum.IntEnum):
    LOW = 1
    HIGH = 2


class Color(enum.Enum):
    RED = 1
    BLUE = 2


@pytest.mark.parametrize(
    ("number1", "number2"),
    [
        (Level.LOW, Level.HIGH),
        (Level.LOW, 2),
        (2, Level.LOW),
        (Level.LOW, 1.5),
        (Color.RED, Color.BLUE),
    ],
)
def test_number_equal_enum_has_no_tolerance(number1: object, number2: object) -> None:
    assert not number_equal(
        number1,  # ty: ignore[invalid-argument-type]
        number2,  # ty: ignore[invalid-argument-type]
        EqualityConfig(tolerance=Tolerance(atol=5.0, rtol=1.0)),
    )


def test_number_equal_same_enum_member_with_tolerance() -> None:
    assert number_equal(Level.LOW, Level.LOW, EqualityConfig(tolerance=Tolerance(atol=5.0)))


@pytest.mark.parametrize(("number1", "number2"), [(True, 1), (1, True), (False, 0.0), (0.0, False)])
def test_number_equal_bool_has_no_tolerance(number1: float, number2: float) -> None:
    assert number_equal(
        number1, number2, EqualityConfig(tolerance=Tolerance(atol=1.0, rtol=1.0))
    ) == (number1 == number2)


def test_number_equal_integer_too_large_for_a_float() -> None:
    assert number_equal(10**400, 10**400 + 1, EqualityConfig(tolerance=Tolerance(atol=1.0)))


def test_number_equal_integer_too_large_for_a_float_and_a_float() -> None:
    assert not number_equal(10**400, 1.0, EqualityConfig(tolerance=Tolerance(atol=1.0)))
