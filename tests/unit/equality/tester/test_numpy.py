from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock, patch

import pytest

from coola.equality import compare
from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    NumpyArrayEqualityTester,
    NumpyMaskedArrayEqualityTester,
    NumpyTimeScalarEqualityTester,
)
from coola.equality.tolerance import Tolerance
from coola.testing.marks import numpy_available
from coola.utils.imports import MissingPackageError, is_numpy_available
from tests.unit.equality.utils import ExamplePair, explaining, messages, run_tester

if is_numpy_available():
    import numpy as np
else:
    np = Mock()


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


NUMPY_ARRAY_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=np.ones(shape=(2, 3), dtype=float), expected=np.ones(shape=(2, 3), dtype=float)
        ),
        id="float dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=np.ones(shape=(2, 3), dtype=int), expected=np.ones(shape=(2, 3), dtype=int)
        ),
        id="int dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=np.array(["polar", "bear", "meow"]), expected=np.array(["polar", "bear", "meow"])
        ),
        id="string dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=np.array(["2005-02-25", "2007-07-13"], dtype=np.datetime64),
            expected=np.array(["2005-02-25", "2007-07-13"], dtype=np.datetime64),
        ),
        id="datetime dtype",
    ),
    pytest.param(ExamplePair(actual=np.ones(shape=6), expected=np.ones(shape=6)), id="1d array"),
    pytest.param(
        ExamplePair(actual=np.ones(shape=(2, 3)), expected=np.ones(shape=(2, 3))), id="2d array"
    ),
]
NUMPY_ARRAY_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=np.ones(shape=(2, 3), dtype=float),
            expected=np.ones(shape=(2, 3), dtype=int),
            expected_message="objects have different data types:",
        ),
        id="different data types",
    ),
    pytest.param(
        ExamplePair(
            actual=np.ones(shape=(2, 3)),
            expected=np.ones(shape=6),
            expected_message="objects have different shapes:",
        ),
        id="different shapes",
    ),
    pytest.param(
        ExamplePair(
            actual=np.ones(shape=(2, 3)),
            expected=np.zeros(shape=(2, 3)),
            expected_message="values are different:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=np.ones(shape=(2, 3)),
            expected="meow",
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]
NUMPY_ARRAY_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(actual=np.ones((2, 3)), expected=np.full((2, 3), 1.5), atol=1.0),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(actual=np.ones((2, 3)), expected=np.full((2, 3), 1.05), atol=0.1),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(actual=np.ones((2, 3)), expected=np.full((2, 3), 1.005), atol=0.01),
        id="atol=0.01",
    ),
    pytest.param(
        ExamplePair(
            actual=np.array(["polar", "bear", "meow"]),
            expected=np.array(["polar", "bear", "meow"]),
            atol=0.01,
        ),
        id="string atol",
    ),
    pytest.param(
        ExamplePair(
            actual=np.array(["2005-02-25", "2007-07-13"], dtype=np.datetime64),
            expected=np.array(["2005-02-25", "2007-07-13"], dtype=np.datetime64),
            atol=0.01,
        ),
        id="datetime atol",
    ),
    # rtol
    pytest.param(
        ExamplePair(actual=np.ones((2, 3)), expected=np.full((2, 3), 1.5), rtol=1.0),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(actual=np.ones((2, 3)), expected=np.full((2, 3), 1.05), rtol=0.1),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(actual=np.ones((2, 3)), expected=np.full((2, 3), 1.005), rtol=0.01),
        id="rtol=0.01",
    ),
    pytest.param(
        ExamplePair(
            actual=np.array(["polar", "bear", "meow"]),
            expected=np.array(["polar", "bear", "meow"]),
            rtol=0.01,
        ),
        id="string rtol",
    ),
    pytest.param(
        ExamplePair(
            actual=np.array(["2005-02-25", "2007-07-13"], dtype=np.datetime64),
            expected=np.array(["2005-02-25", "2007-07-13"], dtype=np.datetime64),
            rtol=0.01,
        ),
        id="datetime rtol",
    ),
]

NUMPY_MASKED_ARRAY_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0], dtype=float),
            expected=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0], dtype=float),
        ),
        id="float dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=np.ma.array(data=[0, 1, 2], mask=[0, 1, 0], dtype=int),
            expected=np.ma.array(data=[0, 1, 2], mask=[0, 1, 0], dtype=int),
        ),
        id="int dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=np.ma.array(data=[0.0, 1.0, 1.2]), expected=np.ma.array(data=[0.0, 1.0, 1.2])
        ),
        id="1d array",
    ),
    pytest.param(
        ExamplePair(
            actual=np.ma.array(data=np.ones(shape=(2, 3)), mask=[[0, 1, 0], [1, 0, 0]]),
            expected=np.ma.array(data=np.ones(shape=(2, 3)), mask=[[0, 1, 0], [1, 0, 0]]),
        ),
        id="2d array",
    ),
]
NUMPY_MASKED_ARRAY_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0], dtype=float),
            expected=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0], dtype=int),
            expected_message="objects have different data types:",
        ),
        id="different data types",
    ),
    pytest.param(
        ExamplePair(
            actual=np.ma.array(data=np.ones(shape=(2, 3)), mask=[[0, 1, 0], [1, 0, 0]]),
            expected=np.ma.array(data=np.ones(shape=(3, 2)), mask=[[0, 1], [1, 0], [0, 0]]),
            expected_message="objects have different shapes:",
        ),
        id="different shapes",
    ),
    pytest.param(
        ExamplePair(
            actual=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0]),
            expected=np.ma.array(data=[0.0, 1.0, 2.0], mask=[0, 1, 0]),
            expected_message="values are different:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0]),
            expected=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 0, 1]),
            expected_message="values are different:",
        ),
        id="different mask",
    ),
    pytest.param(
        ExamplePair(
            actual=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0], fill_value=-1),
            expected=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0], fill_value=42),
            expected_message="numbers are different:",
        ),
        id="different fill_value",
    ),
    pytest.param(
        ExamplePair(
            actual=np.ma.array(data=np.ones(shape=(2, 3)), mask=[[0, 1, 0], [1, 0, 0]]),
            expected=np.ones(shape=(2, 3)),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]
NUMPY_MASKED_ARRAY_EQUAL_TOLERANCE = []

NUMPY_EQUAL = NUMPY_ARRAY_EQUAL + NUMPY_MASKED_ARRAY_EQUAL
NUMPY_NOT_EQUAL = NUMPY_ARRAY_NOT_EQUAL + NUMPY_MASKED_ARRAY_NOT_EQUAL
NUMPY_EQUAL_TOLERANCE = NUMPY_ARRAY_EQUAL_TOLERANCE + NUMPY_MASKED_ARRAY_EQUAL_TOLERANCE


##############################################
#     Tests for NumpyArrayEqualityTester     #
##############################################


@numpy_available
def test_numpy_array_equality_tester_repr() -> None:
    assert repr(NumpyArrayEqualityTester()).startswith("NumpyArrayEqualityTester(")


@numpy_available
def test_numpy_array_equality_tester_str() -> None:
    assert str(NumpyArrayEqualityTester()).startswith("NumpyArrayEqualityTester(")


@numpy_available
def test_numpy_array_equality_tester_equal_true() -> None:
    assert NumpyArrayEqualityTester() == NumpyArrayEqualityTester()


@numpy_available
def test_numpy_array_equality_tester_equal_false_different_type() -> None:
    assert NumpyArrayEqualityTester() != 123


@numpy_available
def test_numpy_array_equality_tester_equal_false_different_type_child() -> None:
    class Child(NumpyArrayEqualityTester): ...

    assert NumpyArrayEqualityTester() != Child()


@numpy_available
def test_numpy_array_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    array = np.ones((2, 3))
    assert run_tester(NumpyArrayEqualityTester(), array, array, config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_EQUAL)
def test_numpy_array_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = NumpyArrayEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_EQUAL)
def test_numpy_array_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = NumpyArrayEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_EQUAL)
def test_numpy_array_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = NumpyArrayEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_EQUAL)
def test_numpy_array_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = NumpyArrayEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_NOT_EQUAL)
def test_numpy_array_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = NumpyArrayEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_NOT_EQUAL)
def test_numpy_array_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = NumpyArrayEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_NOT_EQUAL)
def test_numpy_array_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = NumpyArrayEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_NOT_EQUAL)
def test_numpy_array_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = NumpyArrayEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert messages(config)[0].startswith(example.expected_message)


@numpy_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_numpy_array_equality_tester_objects_are_equal_nan_true(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config = replace(config, equal_nan=equal_nan)
    assert (
        run_tester(
            NumpyArrayEqualityTester(),
            np.array([0.0, np.nan, np.nan, 1.2]),
            np.array([0.0, np.nan, np.nan, 1.2]),
            config,
        )
        == equal_nan
    )


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_EQUAL_TOLERANCE)
def test_numpy_array_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config = replace(config, tolerance=Tolerance(atol=example.atol, rtol=example.rtol))
    assert run_tester(NumpyArrayEqualityTester(), example.actual, example.expected, config)


def test_numpy_array_equality_tester_no_numpy() -> None:
    with (
        patch("coola.utils.imports.optional.is_numpy_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'numpy' package is required but not installed."),
    ):
        NumpyArrayEqualityTester()


####################################################
#     Tests for NumpyMaskedArrayEqualityTester     #
####################################################


@numpy_available
def test_numpy_masked_array_equality_tester_repr() -> None:
    assert repr(NumpyMaskedArrayEqualityTester()).startswith("NumpyMaskedArrayEqualityTester(")


@numpy_available
def test_numpy_masked_array_equality_tester_str() -> None:
    assert str(NumpyMaskedArrayEqualityTester()).startswith("NumpyMaskedArrayEqualityTester(")


@numpy_available
def test_numpy_masked_array_equality_tester_equal_true() -> None:
    assert NumpyMaskedArrayEqualityTester() == NumpyMaskedArrayEqualityTester()


@numpy_available
def test_numpy_masked_array_equality_tester_equal_false_different_type() -> None:
    assert NumpyMaskedArrayEqualityTester() != 123


@numpy_available
def test_numpy_masked_array_equality_tester_equal_false_different_type_child() -> None:
    class Child(NumpyMaskedArrayEqualityTester): ...

    assert NumpyMaskedArrayEqualityTester() != Child()


@numpy_available
def test_numpy_masked_array_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    array = np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0])
    assert run_tester(NumpyMaskedArrayEqualityTester(), array, array, config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_EQUAL)
def test_numpy_masked_array_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = NumpyMaskedArrayEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_EQUAL)
def test_numpy_masked_array_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = NumpyMaskedArrayEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_EQUAL)
def test_numpy_masked_array_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = NumpyMaskedArrayEqualityTester()
    assert run_tester(tester, example.actual, example.expected, config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_EQUAL)
def test_numpy_masked_array_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = NumpyMaskedArrayEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_NOT_EQUAL)
def test_numpy_masked_array_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = NumpyMaskedArrayEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_NOT_EQUAL)
def test_numpy_masked_array_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = NumpyMaskedArrayEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert not messages(config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_NOT_EQUAL)
def test_numpy_masked_array_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = NumpyMaskedArrayEqualityTester()
    assert not run_tester(tester, example.actual, example.expected, config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_NOT_EQUAL)
def test_numpy_masked_array_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config = explaining(config)
    tester = NumpyMaskedArrayEqualityTester()
    run_tester(tester, example.actual, example.expected, config)
    assert messages(config)[0].startswith(example.expected_message)


@numpy_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_numpy_masked_array_equality_tester_objects_are_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config = replace(config, equal_nan=equal_nan)
    assert (
        run_tester(
            NumpyMaskedArrayEqualityTester(),
            np.ma.array(data=[0.0, np.nan, np.nan, 1.2], mask=[0, 1, 0, 1]),
            np.ma.array(data=[0.0, np.nan, np.nan, 1.2], mask=[0, 1, 0, 1]),
            config,
        )
        == equal_nan
    )


def test_numpy_masked_array_equality_tester_no_numpy() -> None:
    with (
        patch("coola.utils.imports.optional.is_numpy_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'numpy' package is required but not installed."),
    ):
        NumpyMaskedArrayEqualityTester()


######################################################
#     Tests for NumpyTimeScalarEqualityTester        #
######################################################


@numpy_available
def test_numpy_time_scalar_equality_tester_equal(config: EqualityConfig) -> None:
    assert run_tester(
        NumpyTimeScalarEqualityTester(), np.timedelta64(1, "s"), np.timedelta64(1, "s"), config
    )


@numpy_available
def test_numpy_time_scalar_equality_tester_different(config: EqualityConfig) -> None:
    assert not run_tester(
        NumpyTimeScalarEqualityTester(), np.timedelta64(1, "s"), np.timedelta64(2, "s"), config
    )


@numpy_available
def test_numpy_time_scalar_equality_tester_different_type(config: EqualityConfig) -> None:
    assert not run_tester(
        NumpyTimeScalarEqualityTester(), np.timedelta64(1, "s"), np.datetime64(1, "s"), config
    )


def test_numpy_time_scalar_equality_tester_no_numpy() -> None:
    with (
        patch("coola.utils.imports.optional.is_numpy_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'numpy' package is required but not installed."),
    ):
        NumpyTimeScalarEqualityTester()


#############################################
#     Tests for the location of a difference #
#############################################


def _object_array(first: object, second: object) -> np.ndarray:
    array = np.empty(2, dtype=object)
    array[0], array[1] = first, second
    return array


@numpy_available
def test_numpy_array_compare_path_is_the_index() -> None:
    expected = np.zeros((2, 3))
    expected[1, 2] = 5.0
    assert compare(np.zeros((2, 3)), expected).path == (1, 2)


@numpy_available
def test_numpy_array_compare_path_of_the_first_difference() -> None:
    expected = np.zeros((2, 3))
    expected[0, 1] = 5.0
    expected[1, 2] = 5.0
    assert compare(np.zeros((2, 3)), expected).path == (0, 1)


@numpy_available
def test_numpy_array_compare_path_is_a_tuple_of_ints() -> None:
    expected = np.zeros(2)
    expected[1] = 5.0
    assert type(compare(np.zeros(2), expected).path[0]) is int


@numpy_available
def test_numpy_array_compare_path_of_a_0d_array() -> None:
    assert compare(np.array(1.0), np.array(2.0)).path == ()


@numpy_available
def test_numpy_array_compare_reason_has_the_two_values() -> None:
    assert (
        compare(np.array([1.0, 2.0]), np.array([1.0, 3.0])).reason
        == "values are different:\n  actual   : 2.0\n  expected : 3.0"
    )


@numpy_available
def test_numpy_array_compare_reason_is_bounded() -> None:
    expected = np.zeros(1_000_000)
    expected[0] = 1.0
    reason = compare(np.zeros(1_000_000), expected).reason
    assert reason is not None
    assert len(reason) < 100


@numpy_available
def test_numpy_array_compare_path_of_a_string_array() -> None:
    assert compare(np.array(["a", "b"]), np.array(["a", "c"])).path == (1,)


@numpy_available
def test_numpy_array_compare_path_with_a_tolerance() -> None:
    assert compare(np.array([1.0, 2.0]), np.array([1.0, 2.5]), atol=0.1).path == (1,)


@numpy_available
def test_numpy_array_compare_path_nan_without_equal_nan() -> None:
    array = np.array([1.0, float("nan")])
    assert compare(array, array.copy()).path == (1,)


@numpy_available
def test_numpy_array_compare_nan_with_equal_nan() -> None:
    array = np.array([1.0, float("nan")])
    assert compare(array, array.copy(), equal_nan=True).equal


@numpy_available
def test_numpy_array_compare_path_of_a_duration_array() -> None:
    assert compare(
        np.array([1, 2], dtype="m8[s]"), np.array([1, 9], dtype="m8[s]"), atol=1.0
    ).path == (1,)


@numpy_available
def test_numpy_array_compare_path_of_a_date_array() -> None:
    assert compare(
        np.array(["2020-01-01", "2020-01-02"], dtype="M8[D]"),
        np.array(["2020-01-01", "2020-01-03"], dtype="M8[D]"),
        atol=1e9,
    ).path == (1,)


@numpy_available
def test_numpy_array_compare_path_of_an_integer_array_with_a_tolerance() -> None:
    assert compare(np.array([1, 2]), np.array([1, 5]), atol=1.0).path == (1,)


@numpy_available
def test_numpy_array_compare_object_arrays_equal() -> None:
    assert compare(_object_array(1.0, np.ones(2)), _object_array(1.0, np.ones(2))).equal


@numpy_available
def test_numpy_array_compare_object_arrays_path_is_the_flat_index_then_the_index() -> None:
    assert compare(_object_array(1.0, np.ones(2)), _object_array(1.0, np.zeros(2))).path == (1, 0)


@numpy_available
def test_numpy_array_compare_object_arrays_equal_nan() -> None:
    assert compare(
        _object_array(float("nan"), 1.0), _object_array(float("nan"), 1.0), equal_nan=True
    ).equal


@numpy_available
def test_numpy_array_compare_object_arrays_nan_without_equal_nan() -> None:
    assert not compare(_object_array(float("nan"), 1.0), _object_array(float("nan"), 1.0)).equal


@numpy_available
def test_numpy_array_compare_object_arrays_tolerance() -> None:
    assert compare(_object_array(1.0, 2.0), _object_array(1.0, 2.05), atol=0.1).equal


@numpy_available
def test_numpy_array_compare_object_arrays_different_shape() -> None:
    assert not compare(np.array([1, 2], dtype=object), np.array([[1, 2]], dtype=object)).equal


@numpy_available
def test_numpy_array_compare_does_not_look_up_the_index_when_not_explaining(
    config: EqualityConfig,
) -> None:
    assert not run_tester(NumpyArrayEqualityTester(), np.zeros(3), np.ones(3), config)


@numpy_available
def test_numpy_masked_array_compare_path_of_the_data() -> None:
    assert compare(np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 9], mask=[0, 1])).path == (
        "data",
        1,
    )


@numpy_available
def test_numpy_masked_array_compare_path_of_the_mask() -> None:
    assert compare(np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 2], mask=[0, 0])).path == (
        "mask",
        1,
    )


@numpy_available
def test_numpy_masked_array_compare_path_of_the_fill_value() -> None:
    assert compare(np.ma.array([1, 2], fill_value=1), np.ma.array([1, 2], fill_value=2)).path == (
        "fill_value",
    )


@numpy_available
def test_numpy_masked_array_compare_path_is_an_attribute_name() -> None:
    result = compare(np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 9], mask=[0, 1]))
    assert result.format_path("actual") == "actual.data[1]"


@numpy_available
def test_numpy_time_scalar_compare_reason() -> None:
    result = compare(np.datetime64("2020-01-01"), np.datetime64("2020-01-02"))
    assert result.reason is not None
    assert result.reason.startswith("numpy time scalars are different:")
