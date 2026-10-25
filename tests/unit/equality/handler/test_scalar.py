from __future__ import annotations

import logging
import math
import numbers

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import (
    FalseHandler,
    NanEqualHandler,
    ScalarEqualHandler,
    TrueHandler,
)
from coola.equality.handler.scalar import complex_are_equal, int_isclose, number_equal
from tests.unit.equality.tester.test_scalar import (
    SCALAR_EQUAL_TOLERANCE,
    SCALAR_NOT_EQUAL_TOLERANCE,
)
from tests.unit.equality.utils import ExamplePair


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


#####################################
#     Tests for NanEqualHandler     #
#####################################


def test_nan_equal_handler_repr() -> None:
    assert repr(NanEqualHandler()) == "NanEqualHandler()"


def test_nan_equal_handler_repr_with_next_handler() -> None:
    assert repr(NanEqualHandler(FalseHandler())) == "NanEqualHandler(next_handler=FalseHandler())"


def test_nan_equal_handler_str() -> None:
    assert str(NanEqualHandler()) == "NanEqualHandler()"


def test_nan_equal_handler_str_with_next_handler() -> None:
    assert str(NanEqualHandler(FalseHandler())) == "NanEqualHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(NanEqualHandler(), NanEqualHandler(), id="without next handler"),
        pytest.param(
            NanEqualHandler(FalseHandler()),
            NanEqualHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_nan_equal_handler_equal_true(handler1: NanEqualHandler, handler2: NanEqualHandler) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            NanEqualHandler(TrueHandler()),
            NanEqualHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            NanEqualHandler(),
            NanEqualHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            NanEqualHandler(FalseHandler()),
            NanEqualHandler(),
            id="other next handler is none",
        ),
        pytest.param(NanEqualHandler(), FalseHandler(), id="different type"),
    ],
)
def test_nan_equal_handler_equal_false(handler1: NanEqualHandler, handler2: object) -> None:
    assert not handler1.equal(handler2)


def test_nan_equal_handler_equal_false_different_type_child() -> None:
    class Child(NanEqualHandler): ...

    assert not NanEqualHandler().equal(Child())


def test_nan_equal_handler_handle_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert NanEqualHandler().handle(float("nan"), float("nan"), config)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (float("nan"), float("nan")),
        (4.2, 4.2),
        (1, 0),
    ],
)
def test_nan_equal_handler_handle_false(
    actual: float, expected: float, config: EqualityConfig
) -> None:
    assert not NanEqualHandler(next_handler=FalseHandler()).handle(actual, expected, config)


def test_nan_equal_handler_handle_without_next_handler(config: EqualityConfig) -> None:
    handler = NanEqualHandler()
    with pytest.raises(RuntimeError, match=r"next handler is not defined"):
        handler.handle(actual=42, expected=42, config=config)


def test_nan_equal_handler_set_next_handler() -> None:
    handler = NanEqualHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_nan_equal_handler_set_next_handler_none() -> None:
    handler = NanEqualHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_nan_equal_handler_set_next_handler_incorrect() -> None:
    handler = NanEqualHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


########################################
#     Tests for ScalarEqualHandler     #
########################################


def test_scalar_equal_handler_repr() -> None:
    assert repr(ScalarEqualHandler()) == "ScalarEqualHandler()"


def test_scalar_equal_handler_repr_with_next_handler() -> None:
    assert (
        repr(ScalarEqualHandler(FalseHandler()))
        == "ScalarEqualHandler(next_handler=FalseHandler())"
    )


def test_scalar_equal_handler_str() -> None:
    assert str(ScalarEqualHandler()) == "ScalarEqualHandler()"


def test_scalar_equal_handler_str_with_next_handler() -> None:
    assert str(ScalarEqualHandler(FalseHandler())) == "ScalarEqualHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(ScalarEqualHandler(), ScalarEqualHandler(), id="without next handler"),
        pytest.param(
            ScalarEqualHandler(FalseHandler()),
            ScalarEqualHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_scalar_equal_handler_equal_true(
    handler1: ScalarEqualHandler, handler2: ScalarEqualHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            ScalarEqualHandler(TrueHandler()),
            ScalarEqualHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            ScalarEqualHandler(),
            ScalarEqualHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            ScalarEqualHandler(FalseHandler()),
            ScalarEqualHandler(),
            id="other next handler is none",
        ),
        pytest.param(ScalarEqualHandler(), FalseHandler(), id="different type"),
    ],
)
def test_scalar_equal_handler_equal_false(handler1: ScalarEqualHandler, handler2: object) -> None:
    assert not handler1.equal(handler2)


def test_scalar_equal_handler_equal_false_different_type_child() -> None:
    class Child(ScalarEqualHandler): ...

    assert not ScalarEqualHandler().equal(Child())


@pytest.mark.parametrize(
    "example",
    [
        pytest.param(ExamplePair(actual=4, expected=4), id="int"),
        pytest.param(ExamplePair(actual=4.2, expected=4.2), id="float"),
        pytest.param(ExamplePair(actual=-1.0, expected=-1.0), id="negative"),
        pytest.param(ExamplePair(actual=float("inf"), expected=float("inf")), id="infinity"),
        pytest.param(ExamplePair(actual=float("-inf"), expected=float("-inf")), id="-infinity"),
    ],
)
def test_scalar_equal_handler_handle_true(example: ExamplePair, config: EqualityConfig) -> None:
    assert ScalarEqualHandler().handle(example.actual, example.expected, config)


@pytest.mark.parametrize(
    "example",
    [
        pytest.param(ExamplePair(actual=0, expected=1), id="different values - int"),
        pytest.param(ExamplePair(actual=4.0, expected=4.2), id="different values - float"),
        pytest.param(ExamplePair(actual=float("inf"), expected=4.2), id="different values - inf"),
        pytest.param(
            ExamplePair(actual=float("inf"), expected=float("-inf")), id="opposite infinity"
        ),
        pytest.param(ExamplePair(actual=float("nan"), expected=1.0), id="one nan"),
        pytest.param(ExamplePair(actual=float("nan"), expected=float("nan")), id="two nans"),
    ],
)
def test_scalar_equal_handler_handle_false(example: ExamplePair, config: EqualityConfig) -> None:
    assert not ScalarEqualHandler().handle(example.actual, example.expected, config)


def test_scalar_equal_handler_handle_false_show_difference(config: EqualityConfig) -> None:
    config.show_difference = True
    assert not ScalarEqualHandler().handle(actual=1.0, expected=2.0, config=config)


def test_scalar_equal_handler_handle_false_show_difference_logs(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    with caplog.at_level(logging.INFO):
        ScalarEqualHandler().handle(actual=1.0, expected=2.0, config=config)
    assert caplog.messages[0].startswith("numbers are different:")


@pytest.mark.parametrize("equal_nan", [True, False])
def test_scalar_equal_handler_handle_equal_nan(config: EqualityConfig, equal_nan: bool) -> None:
    config.equal_nan = equal_nan
    assert not ScalarEqualHandler().handle(float("nan"), float("nan"), config)


@pytest.mark.parametrize("example", SCALAR_EQUAL_TOLERANCE)
def test_scalar_equal_handler_handle_true_tolerance(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert ScalarEqualHandler().handle(example.actual, example.expected, config)


def test_scalar_equal_handler_set_next_handler() -> None:
    handler = ScalarEqualHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_scalar_equal_handler_set_next_handler_none() -> None:
    handler = ScalarEqualHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_scalar_equal_handler_set_next_handler_incorrect() -> None:
    handler = ScalarEqualHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


def test_number_equal_huge_int_rtol() -> None:
    big = 10**400
    assert number_equal(big, big + 1, EqualityConfig(rtol=1e-5))


def test_number_equal_huge_int_rtol_different() -> None:
    big = 10**400
    assert not number_equal(big, 2 * big, EqualityConfig(rtol=1e-5))


def test_number_equal_huge_int_atol_same() -> None:
    big = 10**400
    assert number_equal(big, big, EqualityConfig(atol=1.0))


def test_number_equal_huge_int_atol_different() -> None:
    big = 10**400
    assert not number_equal(big, big + 2, EqualityConfig(atol=1.0))


def test_number_equal_huge_int_atol_close() -> None:
    big = 10**400
    assert number_equal(big, big + 1, EqualityConfig(atol=1.0))


def test_number_equal_large_ints_are_not_rounded_to_float() -> None:
    config = EqualityConfig(rtol=0.0, atol=1e-8)
    assert not number_equal(2**60, 2**60 + 1, config)


def test_number_equal_large_ints_are_equal_to_themselves() -> None:
    config = EqualityConfig(rtol=0.0, atol=1e-8)
    assert number_equal(2**60, 2**60, config)


def test_number_equal_huge_int_and_infinity() -> None:
    assert not number_equal(10**400, float("inf"), EqualityConfig(rtol=1e-5))


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
    assert number_equal(1 + 1j, 1 + 1j, EqualityConfig())


def test_number_equal_complex_exact_different() -> None:
    assert not number_equal(1 + 1j, 1 + 2j, EqualityConfig())


def test_number_equal_complex_atol() -> None:
    assert number_equal(1 + 1j, 1 + 1.05j, EqualityConfig(atol=0.1))


def test_number_equal_complex_rtol() -> None:
    assert number_equal(1 + 1j, 1.05 + 1j, EqualityConfig(rtol=0.1))


def test_number_equal_complex_tolerance_not_equal() -> None:
    assert not number_equal(1 + 1j, 1 + 2j, EqualityConfig(atol=0.1))


def test_number_equal_complex_tolerance_checks_imaginary_part() -> None:
    assert not number_equal(1 + 0j, 1 + 5j, EqualityConfig(atol=0.1))


def test_number_equal_complex_and_real_tolerance() -> None:
    assert number_equal(1 + 0j, 1.05, EqualityConfig(atol=0.1))


def test_number_equal_real_and_complex_tolerance() -> None:
    assert number_equal(1.05, 1 + 0j, EqualityConfig(atol=0.1))


def test_number_equal_complex_with_imaginary_part_and_real_tolerance() -> None:
    assert not number_equal(1 + 1j, 1.0, EqualityConfig(atol=0.1))


def test_nan_equal_handler_handle_complex_nan_parts(config: EqualityConfig) -> None:
    config.equal_nan = True
    handler = NanEqualHandler(next_handler=ScalarEqualHandler())
    assert handler.handle(complex(float("nan"), 1), complex(float("nan"), 1), config)


def test_nan_equal_handler_handle_complex_nan_real_parts_different_imaginary_parts(
    config: EqualityConfig,
) -> None:
    config.equal_nan = True
    handler = NanEqualHandler(next_handler=ScalarEqualHandler())
    assert not handler.handle(complex(float("nan"), 1), complex(float("nan"), 2), config)


def test_nan_equal_handler_handle_complex_nan_imaginary_parts_different_real_parts(
    config: EqualityConfig,
) -> None:
    config.equal_nan = True
    handler = NanEqualHandler(next_handler=ScalarEqualHandler())
    assert not handler.handle(complex(1, float("nan")), complex(2, float("nan")), config)


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
    config = EqualityConfig(equal_nan=True, atol=0.1)
    assert complex_are_equal(complex(NAN, 1), complex(NAN, 1.05), config)


def test_complex_are_equal_tolerance_different() -> None:
    config = EqualityConfig(equal_nan=True, atol=0.1)
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
    assert number_equal(number1, number2, EqualityConfig(atol=0.5, rtol=0.5))


def test_number_equal_atol_and_rtol_add_up_huge_int() -> None:
    big = 10**400
    assert number_equal(big, 3 * big, EqualityConfig(atol=0.5, rtol=1.0))


@pytest.mark.parametrize(
    ("number1", "number2"), [(1.0, 3.5), (3.5, 1.0), (1, 4), (4, 1), (1.0, float("inf"))]
)
def test_number_equal_outside_atol_and_rtol(number1: float, number2: float) -> None:
    assert not number_equal(number1, number2, EqualityConfig(atol=0.5, rtol=0.5))


def test_number_equal_not_a_number_equal() -> None:
    assert number_equal("a", "a", EqualityConfig(atol=0.5))


def test_number_equal_not_a_number_different() -> None:
    assert not number_equal("a", "b", EqualityConfig(atol=0.5))


def test_number_equal_real_without_float_equal() -> None:
    assert number_equal(Unconvertible(), Unconvertible(), EqualityConfig(atol=0.5))


def test_number_equal_real_without_float_different() -> None:
    assert not number_equal(Unconvertible(), Unconvertible(1), EqualityConfig(atol=0.5))


@pytest.mark.parametrize("example", SCALAR_NOT_EQUAL_TOLERANCE)
def test_scalar_equal_handler_handle_false_tolerance(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert not ScalarEqualHandler().handle(example.actual, example.expected, config)


@pytest.mark.parametrize(("number1", "number2"), [(True, 1), (1, True), (False, 0.0), (0.0, False)])
def test_number_equal_bool_has_no_tolerance(number1: float, number2: float) -> None:
    assert number_equal(number1, number2, EqualityConfig(atol=1.0, rtol=1.0)) == (
        number1 == number2
    )
