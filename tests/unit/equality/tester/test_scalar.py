from __future__ import annotations

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import ScalarEqualityTester
from tests.unit.equality.utils import ExamplePair, messages, run_tester


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


FLOAT_EQUAL = [
    pytest.param(ExamplePair(actual=4.2, expected=4.2), id="positive"),
    pytest.param(ExamplePair(actual=0.0, expected=0.0), id="zero"),
    pytest.param(ExamplePair(actual=-4.2, expected=-4.2), id="negative"),
    pytest.param(ExamplePair(actual=float("inf"), expected=float("inf")), id="infinity"),
    pytest.param(ExamplePair(actual=float("-inf"), expected=float("-inf")), id="-infinity"),
]

FLOAT_NOT_EQUAL = [
    pytest.param(
        ExamplePair(actual=1, expected=2, expected_message="numbers are different:"),
        id="different values - int",
    ),
    pytest.param(
        ExamplePair(actual=1.0, expected=2.0, expected_message="numbers are different:"),
        id="different values - float",
    ),
    pytest.param(
        ExamplePair(actual=4.2, expected="meow", expected_message="objects have different types:"),
        id="different types",
    ),
]

SCALAR_EQUAL = FLOAT_EQUAL
SCALAR_NOT_EQUAL = FLOAT_NOT_EQUAL

SCALAR_EQUAL_TOLERANCE = [
    # atol
    pytest.param(ExamplePair(actual=0, expected=1, atol=1.0), id="integer 0 atol=1"),
    pytest.param(ExamplePair(actual=1, expected=0, atol=1.0), id="integer 1 atol=1"),
    pytest.param(ExamplePair(actual=1, expected=2, atol=1.0), id="integer 2 atol=1"),
    pytest.param(ExamplePair(actual=1, expected=5, atol=10.0), id="integer 1 atol=10"),
    pytest.param(ExamplePair(actual=1.0, expected=1.0001, atol=1e-3), id="float + atol=1e-3"),
    pytest.param(ExamplePair(actual=1.0, expected=0.9999, atol=1e-3), id="float - atol=1e-3"),
    # rtol
    pytest.param(ExamplePair(actual=0, expected=1, rtol=1.0), id="integer 0 rtol=1"),
    pytest.param(ExamplePair(actual=1, expected=0, rtol=1.0), id="integer 1 rtol=1"),
    pytest.param(ExamplePair(actual=1, expected=2, rtol=1.0), id="integer 2 rtol=1"),
    pytest.param(ExamplePair(actual=1, expected=5, rtol=10.0), id="integer 1 rtol=10"),
    pytest.param(ExamplePair(actual=1.0, expected=1.0001, rtol=1e-3), id="float + rtol=1e-3"),
    pytest.param(ExamplePair(actual=1.0, expected=0.9999, rtol=1e-3), id="float - rtol=1e-3"),
]

SCALAR_NOT_EQUAL_TOLERANCE = [
    # A boolean is a label, not a quantity: no tolerance.
    pytest.param(ExamplePair(actual=True, expected=False, atol=1.0), id="bool - atol=1"),
    pytest.param(ExamplePair(actual=False, expected=True, atol=1.0), id="bool reversed - atol=1"),
    pytest.param(ExamplePair(actual=True, expected=False, rtol=1.0), id="bool - rtol=1"),
]


##########################################
#     Tests for ScalarEqualityTester     #
##########################################


def test_scalar_equality_tester_repr() -> None:
    assert repr(ScalarEqualityTester()).startswith("ScalarEqualityTester(")


def test_scalar_equality_tester_str() -> None:
    assert str(ScalarEqualityTester()).startswith("ScalarEqualityTester(")


def test_scalar_equality_tester_equal_true() -> None:
    assert ScalarEqualityTester().equal(ScalarEqualityTester())


def test_scalar_equality_tester_equal_false_different_type() -> None:
    assert not ScalarEqualityTester().equal(123)


def test_scalar_equality_tester_equal_false_different_type_child() -> None:
    class Child(ScalarEqualityTester): ...

    assert not ScalarEqualityTester().equal(Child())


def test_scalar_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    x = 4.2
    assert run_tester(ScalarEqualityTester(), x, x, config)


@pytest.mark.parametrize("example", FLOAT_EQUAL)
def test_scalar_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = ScalarEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", FLOAT_EQUAL)
def test_scalar_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = ScalarEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", FLOAT_EQUAL)
def test_scalar_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = ScalarEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", FLOAT_EQUAL)
def test_scalar_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = ScalarEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", FLOAT_NOT_EQUAL)
def test_scalar_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = ScalarEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", FLOAT_NOT_EQUAL)
def test_scalar_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = ScalarEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@pytest.mark.parametrize("example", FLOAT_NOT_EQUAL)
def test_scalar_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = ScalarEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@pytest.mark.parametrize("example", FLOAT_NOT_EQUAL)
def test_scalar_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = ScalarEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert messages(config)[0].startswith(example.expected_message)


@pytest.mark.parametrize("equal_nan", [False, True])
def test_scalar_equality_tester_objects_are_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert run_tester(ScalarEqualityTester(), float("nan"), float("nan"), config) == equal_nan


@pytest.mark.parametrize("example", SCALAR_EQUAL_TOLERANCE)
def test_scalar_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert run_tester(ScalarEqualityTester(), example.actual, example.expected, config)


@pytest.mark.parametrize("example", SCALAR_NOT_EQUAL_TOLERANCE)
def test_scalar_equality_tester_objects_are_equal_false_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert not run_tester(ScalarEqualityTester(), example.actual, example.expected, config)


def test_scalar_equality_tester_objects_are_equal_true_bool_tolerance(
    config: EqualityConfig,
) -> None:
    config.atol = 1.0
    assert run_tester(ScalarEqualityTester(), True, True, config)


def test_scalar_equality_tester_objects_are_equal_complex_nan_parts() -> None:
    assert run_tester(
        ScalarEqualityTester(),
        complex(float("nan"), 1),
        complex(float("nan"), 1),
        EqualityConfig(equal_nan=True),
    )


def test_scalar_equality_tester_objects_are_equal_complex_nan_real_parts_different_imag() -> None:
    assert not run_tester(
        ScalarEqualityTester(),
        complex(float("nan"), 1),
        complex(float("nan"), 2),
        EqualityConfig(equal_nan=True),
    )


def test_scalar_equality_tester_objects_are_equal_complex_nan_imag_parts_different_real() -> None:
    assert not run_tester(
        ScalarEqualityTester(),
        complex(1, float("nan")),
        complex(2, float("nan")),
        EqualityConfig(equal_nan=True),
    )
