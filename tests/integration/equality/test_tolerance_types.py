r"""Check that the tolerance and ``equal_nan`` apply to the common
numeric types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from coola.equality import objects_are_allclose, objects_are_equal
from coola.testing.fixtures import numpy_available

NAN = float("nan")


@dataclass
class Point:
    x: float
    y: float = 0.0


@dataclass
class WithArray:
    data: np.ndarray


@dataclass
class Ignored:
    x: float
    note: str = field(default="", compare=False)


@dataclass(eq=False)
class CustomEq:
    x: float

    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = None  # type: ignore[assignment]


@numpy_available
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_numpy_float_scalar_tolerance(dtype: type) -> None:
    assert objects_are_allclose(dtype(1.0), dtype(1.01), atol=0.1)


@numpy_available
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_numpy_float_scalar_tolerance_different(dtype: type) -> None:
    assert not objects_are_allclose(dtype(1.0), dtype(2.0), atol=0.1)


@numpy_available
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_numpy_float_scalar_equal_is_exact(dtype: type) -> None:
    assert not objects_are_equal(dtype(1.0), dtype(1.01))


@numpy_available
@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint32])
def test_numpy_int_scalar_tolerance(dtype: type) -> None:
    assert objects_are_allclose(dtype(10), dtype(11), atol=2)


@numpy_available
@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint32])
def test_numpy_int_scalar_tolerance_different(dtype: type) -> None:
    assert not objects_are_allclose(dtype(10), dtype(20), atol=2)


@numpy_available
@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint32])
def test_numpy_int_scalar_equal(dtype: type) -> None:
    assert objects_are_equal(dtype(10), dtype(10))


@numpy_available
def test_numpy_scalar_returns_true() -> None:
    assert objects_are_equal(np.float32(1.0), np.float32(1.0)) is True


@numpy_available
def test_numpy_scalar_returns_false() -> None:
    assert objects_are_equal(np.float32(1.0), np.float32(2.0)) is False


@numpy_available
def test_numpy_scalar_nan_is_not_equal() -> None:
    assert not objects_are_equal(np.float32("nan"), np.float32("nan"))


@numpy_available
def test_numpy_scalar_equal_nan() -> None:
    assert objects_are_equal(np.float32("nan"), np.float32("nan"), equal_nan=True)


@numpy_available
def test_numpy_scalar_different_types() -> None:
    assert not objects_are_equal(np.float32(1.0), np.float64(1.0))


@numpy_available
def test_numpy_complex_scalar_tolerance() -> None:
    assert objects_are_allclose(np.complex64(1 + 1j), np.complex64(1.01 + 1j), atol=0.1)


@numpy_available
def test_numpy_complex_scalar_tolerance_different() -> None:
    assert not objects_are_allclose(np.complex64(1 + 1j), np.complex64(1 + 2j), atol=0.1)


def test_complex_exact() -> None:
    assert objects_are_equal(1 + 1j, 1 + 1j)


def test_complex_exact_different() -> None:
    assert not objects_are_equal(1 + 1j, 1 + 1.01j)


def test_complex_tolerance() -> None:
    assert objects_are_allclose(1 + 1j, 1 + 1.01j, atol=0.1)


def test_complex_tolerance_different() -> None:
    assert not objects_are_allclose(1 + 1j, 1 + 2j, atol=0.1)


def test_complex_imaginary_part_is_compared() -> None:
    assert not objects_are_allclose(1 + 0j, 1 + 5j, atol=0.1)


def test_complex_nan_is_not_equal() -> None:
    assert not objects_are_equal(complex(NAN, 0), complex(NAN, 0))


def test_complex_equal_nan() -> None:
    assert objects_are_equal(complex(NAN, 0), complex(NAN, 0), equal_nan=True)


def test_complex_equal_nan_real_and_imaginary_parts() -> None:
    assert objects_are_equal(complex(NAN, NAN), complex(NAN, NAN), equal_nan=True)


def test_complex_not_equal_nan_vs_number() -> None:
    assert not objects_are_equal(complex(NAN, 0), 1 + 0j, equal_nan=True)


def test_dataclass_equal() -> None:
    assert objects_are_equal(Point(1.0, 2.0), Point(1.0, 2.0)) is True


def test_dataclass_not_equal() -> None:
    assert objects_are_equal(Point(1.0, 2.0), Point(1.0, 3.0)) is False


def test_dataclass_different_types() -> None:
    assert not objects_are_equal(Point(1.0), Ignored(1.0))


def test_dataclass_tolerance() -> None:
    assert objects_are_allclose(Point(1.0), Point(1.0000001), atol=1e-3)


def test_dataclass_tolerance_different() -> None:
    assert not objects_are_allclose(Point(1.0), Point(1.1), atol=1e-3)


def test_dataclass_nan_is_not_equal() -> None:
    assert not objects_are_equal(Point(float("nan")), Point(float("nan")))


def test_dataclass_equal_nan() -> None:
    assert objects_are_equal(Point(float("nan")), Point(float("nan")), equal_nan=True)


def test_dataclass_nested() -> None:
    assert objects_are_allclose({"p": [Point(1.0)]}, {"p": [Point(1.0000001)]}, atol=1e-3)


def test_dataclass_ignores_compare_false_fields() -> None:
    assert objects_are_equal(Ignored(1.0, "a"), Ignored(1.0, "b"))


@numpy_available
def test_dataclass_with_array() -> None:
    assert objects_are_equal(WithArray(np.ones(3)), WithArray(np.ones(3))) is True


@numpy_available
def test_dataclass_with_array_different() -> None:
    assert objects_are_equal(WithArray(np.ones(3)), WithArray(np.zeros(3))) is False


@numpy_available
def test_dataclass_with_array_tolerance() -> None:
    assert objects_are_allclose(WithArray(np.ones(3)), WithArray(np.ones(3) + 1e-9))


def test_dataclass_with_custom_eq_keeps_its_eq() -> None:
    assert objects_are_equal(CustomEq(1.0), CustomEq(2.0))


def test_dataclass_show_difference() -> None:
    assert not objects_are_equal(Point(1.0, 2.0), Point(1.0, 3.0), show_difference=True)


def test_dataclass_show_difference_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO"):
        objects_are_equal(Point(1.0, 2.0), Point(1.0, 3.0), show_difference=True)
    assert "objects have different y" in caplog.text


def test_dataclass_class_is_not_an_instance() -> None:
    assert objects_are_equal(Point, Point)


#############################################################
#     Tests for the types compared with their difference    #
#############################################################


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(Decimal("1.0"), Decimal("1.00001"), id="decimal"),
        pytest.param(Fraction(1, 3), Fraction(333333, 1000000), id="fraction"),
        pytest.param(timedelta(seconds=1), timedelta(seconds=1.0001), id="timedelta"),
    ],
)
def test_objects_are_allclose_difference_types_atol(actual: object, expected: object) -> None:
    assert objects_are_allclose(actual, expected, atol=1e-3)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(Decimal("1.0"), Decimal("1.00001"), id="decimal"),
        pytest.param(Fraction(1, 3), Fraction(333333, 1000000), id="fraction"),
        pytest.param(timedelta(seconds=1), timedelta(seconds=1.0001), id="timedelta"),
    ],
)
def test_objects_are_equal_difference_types_ignore_the_default_tolerance(
    actual: object, expected: object
) -> None:
    assert not objects_are_equal(actual, expected)


def test_objects_are_allclose_difference_types_in_a_list() -> None:
    assert objects_are_allclose([Decimal("1.0")], [Decimal("1.00001")], atol=1e-3)


def test_objects_are_equal_decimal_nan_equal_nan() -> None:
    assert objects_are_equal(Decimal("NaN"), Decimal("NaN"), equal_nan=True)


########################################
#     Tests for the tolerance rule     #
########################################

# |1 - 3| = 2 <= 0.5 + 0.5 * 3: the absolute and the relative tolerances add up.
PAIRS_BY_TYPE = {
    "float": (1.0, 3.0),
    "int": (1, 3),
    "complex": (complex(1, 0), complex(3, 0)),
    "Decimal": (Decimal(1), Decimal(3)),
    "Fraction": (Fraction(1), Fraction(3)),
    "timedelta": (timedelta(seconds=1), timedelta(seconds=3)),
    "numpy float64": (np.float64(1), np.float64(3)),
    "numpy array": (np.array([1.0]), np.array([3.0])),
    "list": ([1.0], [3.0]),
}


@pytest.mark.parametrize("pair", PAIRS_BY_TYPE.values(), ids=PAIRS_BY_TYPE.keys())
def test_allclose_atol_and_rtol_add_up(pair: tuple) -> None:
    assert objects_are_allclose(*pair, atol=0.5, rtol=0.5)


@pytest.mark.parametrize("pair", PAIRS_BY_TYPE.values(), ids=PAIRS_BY_TYPE.keys())
def test_allclose_atol_and_rtol_add_up_is_symmetric(pair: tuple) -> None:
    assert objects_are_allclose(*reversed(pair), atol=0.5, rtol=0.5)


############################################
#     Tests for NumPy time scalars         #
############################################


@numpy_available
def test_objects_are_allclose_timedelta64_scalar() -> None:
    assert objects_are_allclose(np.timedelta64(1, "s"), np.timedelta64(1, "s"))


@numpy_available
def test_objects_are_allclose_timedelta64_scalar_in_list() -> None:
    assert objects_are_allclose([np.timedelta64(1, "s")], [np.timedelta64(1, "s")])


@numpy_available
def test_objects_are_equal_timedelta64_nat_equal_nan() -> None:
    assert objects_are_equal(np.timedelta64("NaT", "s"), np.timedelta64("NaT", "s"), equal_nan=True)


@numpy_available
def test_objects_are_equal_timedelta64_scalar() -> None:
    assert objects_are_equal(np.timedelta64(1, "s"), np.timedelta64(1, "s"))


@numpy_available
def test_objects_are_equal_timedelta64_scalar_different() -> None:
    assert not objects_are_equal(np.timedelta64(1, "s"), np.timedelta64(2, "s"))


########################################
#     Tests for the option types       #
########################################


@pytest.mark.parametrize("value", [True, 1.5])
def test_objects_are_equal_max_depth_wrong_type(value: object) -> None:
    with pytest.raises(TypeError, match="max_depth must be an int"):
        objects_are_equal([1], [1], max_depth=value)


def test_objects_are_allclose_atol_bool() -> None:
    with pytest.raises(TypeError, match="atol must be a real number"):
        objects_are_allclose(1.0, 1.0, atol=True)
