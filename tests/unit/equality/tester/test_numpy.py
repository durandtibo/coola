from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    NumpyArrayEqualityTester,
    NumpyMaskedArrayEqualityTester,
    NumpyTimeScalarEqualityTester,
)
from coola.testing.fixtures import numpy_available
from coola.utils.imports import MissingPackageError, is_numpy_available
from tests.unit.equality.utils import ExamplePair, messages

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
            expected_message="numpy.ndarrays are different:",
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
            expected_message="objects have different data:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            actual=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0]),
            expected=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 0, 1]),
            expected_message="objects have different mask:",
        ),
        id="different mask",
    ),
    pytest.param(
        ExamplePair(
            actual=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0], fill_value=-1),
            expected=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0], fill_value=42),
            expected_message="objects have different fill_value:",
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
    assert NumpyArrayEqualityTester().equal(NumpyArrayEqualityTester())


@numpy_available
def test_numpy_array_equality_tester_equal_false_different_type() -> None:
    assert not NumpyArrayEqualityTester().equal(123)


@numpy_available
def test_numpy_array_equality_tester_equal_false_different_type_child() -> None:
    class Child(NumpyArrayEqualityTester): ...

    assert not NumpyArrayEqualityTester().equal(Child())


@numpy_available
def test_numpy_array_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    array = np.ones((2, 3))
    assert NumpyArrayEqualityTester().objects_are_equal(array, array, config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_EQUAL)
def test_numpy_array_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = NumpyArrayEqualityTester()
    assert tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_EQUAL)
def test_numpy_array_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = NumpyArrayEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_EQUAL)
def test_numpy_array_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = NumpyArrayEqualityTester()
    assert tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_EQUAL)
def test_numpy_array_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = NumpyArrayEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_NOT_EQUAL)
def test_numpy_array_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = NumpyArrayEqualityTester()
    assert not tester.objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_NOT_EQUAL)
def test_numpy_array_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = NumpyArrayEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_NOT_EQUAL)
def test_numpy_array_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = NumpyArrayEqualityTester()
    assert not tester.objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_NOT_EQUAL)
def test_numpy_array_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = NumpyArrayEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert messages(config)[-1].startswith(example.expected_message)


@numpy_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_numpy_array_equality_tester_objects_are_equal_nan_true(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        NumpyArrayEqualityTester().objects_are_equal(
            actual=np.array([0.0, np.nan, np.nan, 1.2]),
            expected=np.array([0.0, np.nan, np.nan, 1.2]),
            config=config,
        )
        == equal_nan
    )


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_EQUAL_TOLERANCE)
def test_numpy_array_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert NumpyArrayEqualityTester().objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


def test_numpy_array_equality_tester_no_numpy() -> None:
    with (
        patch("coola.utils.imports.numpy.is_numpy_available", lambda: False),
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
    assert NumpyMaskedArrayEqualityTester().equal(NumpyMaskedArrayEqualityTester())


@numpy_available
def test_numpy_masked_array_equality_tester_equal_false_different_type() -> None:
    assert not NumpyMaskedArrayEqualityTester().equal(123)


@numpy_available
def test_numpy_masked_array_equality_tester_equal_false_different_type_child() -> None:
    class Child(NumpyMaskedArrayEqualityTester): ...

    assert not NumpyMaskedArrayEqualityTester().equal(Child())


@numpy_available
def test_numpy_masked_array_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    array = np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0])
    assert NumpyMaskedArrayEqualityTester().objects_are_equal(array, array, config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_EQUAL)
def test_numpy_masked_array_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = NumpyMaskedArrayEqualityTester()
    assert tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_EQUAL)
def test_numpy_masked_array_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = NumpyMaskedArrayEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_EQUAL)
def test_numpy_masked_array_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = NumpyMaskedArrayEqualityTester()
    assert tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_EQUAL)
def test_numpy_masked_array_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = NumpyMaskedArrayEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_NOT_EQUAL)
def test_numpy_masked_array_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = NumpyMaskedArrayEqualityTester()
    assert not tester.objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_NOT_EQUAL)
def test_numpy_masked_array_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = NumpyMaskedArrayEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_NOT_EQUAL)
def test_numpy_masked_array_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = NumpyMaskedArrayEqualityTester()
    assert not tester.objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_NOT_EQUAL)
def test_numpy_masked_array_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = NumpyMaskedArrayEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert messages(config)[-1].startswith(example.expected_message)


@numpy_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_numpy_masked_array_equality_tester_objects_are_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        NumpyMaskedArrayEqualityTester().objects_are_equal(
            actual=np.ma.array(data=[0.0, np.nan, np.nan, 1.2], mask=[0, 1, 0, 1]),
            expected=np.ma.array(data=[0.0, np.nan, np.nan, 1.2], mask=[0, 1, 0, 1]),
            config=config,
        )
        == equal_nan
    )


def test_numpy_masked_array_equality_tester_no_numpy() -> None:
    with (
        patch("coola.utils.imports.numpy.is_numpy_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'numpy' package is required but not installed."),
    ):
        NumpyMaskedArrayEqualityTester()


######################################################
#     Tests for NumpyTimeScalarEqualityTester        #
######################################################


@numpy_available
def test_numpy_time_scalar_equality_tester_equal(config: EqualityConfig) -> None:
    assert NumpyTimeScalarEqualityTester().objects_are_equal(
        np.timedelta64(1, "s"), np.timedelta64(1, "s"), config
    )


@numpy_available
def test_numpy_time_scalar_equality_tester_different(config: EqualityConfig) -> None:
    assert not NumpyTimeScalarEqualityTester().objects_are_equal(
        np.timedelta64(1, "s"), np.timedelta64(2, "s"), config
    )


@numpy_available
def test_numpy_time_scalar_equality_tester_different_type(config: EqualityConfig) -> None:
    assert not NumpyTimeScalarEqualityTester().objects_are_equal(
        np.timedelta64(1, "s"), np.datetime64(1, "s"), config
    )


def test_numpy_time_scalar_equality_tester_no_numpy() -> None:
    with (
        patch("coola.utils.imports.numpy.is_numpy_available", lambda: False),
        pytest.raises(MissingPackageError, match=r"'numpy' package is required but not installed."),
    ):
        NumpyTimeScalarEqualityTester()
