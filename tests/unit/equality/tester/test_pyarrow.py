from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import PyarrowEqualityTester
from coola.testing.fixtures import pyarrow_available
from coola.utils.imports import MissingPackageError, is_pyarrow_available
from tests.unit.equality.utils import ExamplePair, messages

if is_pyarrow_available():
    import pyarrow as pa
else:
    pa = Mock()


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


PYARROW_ARRAY_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        ),
        id="array float dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1, 2, 3], type=pa.int64()),
            expected=pa.array([1, 2, 3], type=pa.int64()),
        ),
        id="array int dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([True, False, True], type=pa.bool_()),
            expected=pa.array([True, False, True], type=pa.bool_()),
        ),
        id="array bool dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array(["polar", "bear", "meow"], type=pa.string()),
            expected=pa.array(["polar", "bear", "meow"], type=pa.string()),
        ),
        id="array string dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array(
                [
                    datetime(year=2012, month=1, day=1, tzinfo=UTC),
                    datetime(year=2012, month=3, day=4, tzinfo=UTC),
                ],
                type=pa.date64(),
            ),
            expected=pa.array(
                [
                    datetime(year=2012, month=1, day=1, tzinfo=UTC),
                    datetime(year=2012, month=3, day=4, tzinfo=UTC),
                ],
                type=pa.date64(),
            ),
        ),
        id="array date64 dtype",
    ),
]
PYARROW_ARRAY_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1, 2, 3], type=pa.int64()),
            expected_message="objects have different types:",
        ),
        id="array different data types",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 4.0], type=pa.float64()),
            expected_message="objects are different:",
        ),
        id="array different values",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected="meow",
            expected_message="objects have different types:",
        ),
        id="array different types",
    ),
]
PYARROW_ARRAY_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            atol=1.0,
        ),
        id="array atol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            atol=0.1,
        ),
        id="array atol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            atol=0.01,
        ),
        id="array atol=0.01",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array(["polar", "bear", "meow"], type=pa.string()),
            expected=pa.array(["polar", "bear", "meow"], type=pa.string()),
            atol=0.01,
        ),
        id="array string atol",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array(
                [
                    datetime(year=2012, month=1, day=1, tzinfo=UTC),
                    datetime(year=2012, month=3, day=4, tzinfo=UTC),
                ],
                type=pa.date64(),
            ),
            expected=pa.array(
                [
                    datetime(year=2012, month=1, day=1, tzinfo=UTC),
                    datetime(year=2012, month=3, day=4, tzinfo=UTC),
                ],
                type=pa.date64(),
            ),
            atol=0.01,
        ),
        id="array date64 atol",
    ),
    # rtol
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            rtol=1.0,
        ),
        id="array rtol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            rtol=0.1,
        ),
        id="array rtol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            rtol=0.01,
        ),
        id="array rtol=0.01",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array(["polar", "bear", "meow"], type=pa.string()),
            expected=pa.array(["polar", "bear", "meow"], type=pa.string()),
            rtol=0.01,
        ),
        id="array string rtol",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array(
                [
                    datetime(year=2012, month=1, day=1, tzinfo=UTC),
                    datetime(year=2012, month=3, day=4, tzinfo=UTC),
                ],
                type=pa.date64(),
            ),
            expected=pa.array(
                [
                    datetime(year=2012, month=1, day=1, tzinfo=UTC),
                    datetime(year=2012, month=3, day=4, tzinfo=UTC),
                ],
                type=pa.date64(),
            ),
            rtol=0.01,
        ),
        id="array date64 rtol",
    ),
]
PYARROW_ARRAY_NOT_EQUAL_TOLERANCE = [
    # The arrays are different without tolerance and equal with it.
    # atol
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected=pa.array([1.5, 1.5, 1.5], type=pa.float64()),
            atol=1.0,
        ),
        id="array atol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected=pa.array([1.05, 1.05, 1.05], type=pa.float64()),
            atol=0.1,
        ),
        id="array atol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected=pa.array([1.005, 1.005, 1.005], type=pa.float64()),
            atol=0.01,
        ),
        id="array atol=0.01",
    ),
    # rtol
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected=pa.array([1.5, 1.5, 1.5], type=pa.float64()),
            rtol=1.0,
        ),
        id="array rtol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected=pa.array([1.05, 1.05, 1.05], type=pa.float64()),
            rtol=0.1,
        ),
        id="array rtol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected=pa.array([1.005, 1.005, 1.005], type=pa.float64()),
            rtol=0.01,
        ),
        id="array rtol=0.01",
    ),
]

PYARROW_TABLE_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pa.table({"nums": [1, 2, 3, 4, 5]}, schema=pa.schema({"nums": pa.int64()})),
            expected=pa.table({"nums": [1, 2, 3, 4, 5]}, schema=pa.schema({"nums": pa.int64()})),
        ),
        id="table 1 column",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.table(
                {"nums": [1, 2, 3, 4, 5], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums": pa.int64(), "chars": pa.string()}),
            ),
            expected=pa.table(
                {"nums": [1, 2, 3, 4, 5], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums": pa.int64(), "chars": pa.string()}),
            ),
        ),
        id="table 2 columns",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.table(
                {
                    "nums": [1, 2, 3, 4, 5],
                    "float": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "chars": ["a", "b", "c", "d", "e"],
                },
                schema=pa.schema({"nums": pa.int64(), "float": pa.float64(), "chars": pa.string()}),
            ),
            expected=pa.table(
                {
                    "nums": [1, 2, 3, 4, 5],
                    "float": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "chars": ["a", "b", "c", "d", "e"],
                },
                schema=pa.schema({"nums": pa.int64(), "float": pa.float64(), "chars": pa.string()}),
            ),
        ),
        id="table 3 columns",
    ),
]
PYARROW_TABLE_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=pa.table(
                {"nums": [1, 2, 3, 4, 5], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums": pa.int64(), "chars": pa.string()}),
            ),
            expected=pa.table(
                {"nums": [1.0, 2.0, 3.0, 4.0, 5.0], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums": pa.float64(), "chars": pa.string()}),
            ),
            expected_message="objects are different:",
        ),
        id="table different dtypes",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.table(
                {"nums": [1, 2, 3, 4, 5], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums": pa.int64(), "chars": pa.string()}),
            ),
            expected=pa.table(
                {"nums": [1, 2, 3, 4, 6], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums": pa.int64(), "chars": pa.string()}),
            ),
            expected_message="objects are different:",
        ),
        id="table different values",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.table(
                {"nums": [1, 2, 3, 4, 5], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums": pa.int64(), "chars": pa.string()}),
            ),
            expected=pa.table(
                {"nums2": [1, 2, 3, 4, 5], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums2": pa.int64(), "chars": pa.string()}),
            ),
            expected_message="objects are different:",
        ),
        id="table different columns",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.table(
                {"nums": [1, 2, 3, 4, 5], "chars": ["a", "b", "c", "d", "e"]},
                schema=pa.schema({"nums": pa.int64(), "chars": pa.string()}),
            ),
            expected=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected_message="objects have different types:",
        ),
        id="table different types",
    ),
]

PYARROW_EQUAL = PYARROW_ARRAY_EQUAL + PYARROW_TABLE_EQUAL
PYARROW_NOT_EQUAL = PYARROW_ARRAY_NOT_EQUAL + PYARROW_TABLE_NOT_EQUAL
PYARROW_EQUAL_TOLERANCE = PYARROW_ARRAY_EQUAL_TOLERANCE + PYARROW_ARRAY_NOT_EQUAL_TOLERANCE
PYARROW_NOT_EQUAL_TOLERANCE = [
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected=pa.array([1.5, 1.5, 1.5], type=pa.float64()),
            atol=0.1,
        ),
        id="array atol too small",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            expected=pa.array([1.5, 1.5, 1.5], type=pa.float64()),
            rtol=0.1,
        ),
        id="array rtol too small",
    ),
]


###########################################
#     Tests for PyarrowEqualityTester     #
###########################################


@pyarrow_available
def test_pyarrow_equality_tester_repr() -> None:
    assert str(PyarrowEqualityTester()).startswith("PyarrowEqualityTester(")


@pyarrow_available
def test_pyarrow_equality_tester_str() -> None:
    assert str(PyarrowEqualityTester()).startswith("PyarrowEqualityTester(")


@pyarrow_available
def test_pyarrow_equality_tester_equal_true() -> None:
    assert PyarrowEqualityTester().equal(PyarrowEqualityTester())


@pyarrow_available
def test_pyarrow_equality_tester_equal_false_different_type() -> None:
    assert not PyarrowEqualityTester().equal(123)


@pyarrow_available
def test_pyarrow_equality_tester_equal_false_different_type_child() -> None:
    class Child(PyarrowEqualityTester): ...

    assert not PyarrowEqualityTester().equal(Child())


@pyarrow_available
def test_pyarrow_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    array = pa.array([1.0, 2.0, 3.0], type=pa.float64())
    assert PyarrowEqualityTester().objects_are_equal(array, array, config)


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_EQUAL)
def test_pyarrow_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = PyarrowEqualityTester()
    assert tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_EQUAL)
def test_pyarrow_equality_tester_objects_are_equal_true_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = PyarrowEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_EQUAL)
def test_pyarrow_equality_tester_objects_are_equal_true_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = PyarrowEqualityTester()
    assert tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_EQUAL)
def test_pyarrow_equality_tester_objects_are_equal_true_explaining_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = PyarrowEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_NOT_EQUAL)
def test_pyarrow_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = PyarrowEqualityTester()
    assert not tester.objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_NOT_EQUAL)
def test_pyarrow_equality_tester_objects_are_equal_false_does_not_log(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    tester = PyarrowEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert not messages(config)


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_NOT_EQUAL)
def test_pyarrow_equality_tester_objects_are_equal_false_explaining(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = PyarrowEqualityTester()
    assert not tester.objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_NOT_EQUAL)
def test_pyarrow_equality_tester_objects_are_equal_false_explaining_reports(
    example: ExamplePair,
    config: EqualityConfig,
) -> None:
    config.reports = []
    tester = PyarrowEqualityTester()
    tester.objects_are_equal(actual=example.actual, expected=example.expected, config=config)
    assert messages(config)[0].startswith(example.expected_message)


@pyarrow_available
def test_pyarrow_equality_tester_objects_are_equal_nan_false(config: EqualityConfig) -> None:
    assert not PyarrowEqualityTester().objects_are_equal(
        actual=pa.array([0.0, float("nan"), float("nan"), 1.2]),
        expected=pa.array([0.0, float("nan"), float("nan"), 1.2]),
        config=config,
    )


@pyarrow_available
def test_pyarrow_equality_tester_objects_are_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert PyarrowEqualityTester().objects_are_equal(
        actual=pa.array([0.0, float("nan"), float("nan"), 1.2]),
        expected=pa.array([0.0, float("nan"), float("nan"), 1.2]),
        config=config,
    )


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_EQUAL_TOLERANCE)
def test_pyarrow_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PyarrowEqualityTester().objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


def test_pyarrow_equality_tester_no_pyarrow() -> None:
    with (
        patch("coola.utils.imports.pyarrow.is_pyarrow_available", lambda: False),
        pytest.raises(
            MissingPackageError, match=r"'pyarrow' package is required but not installed."
        ),
    ):
        PyarrowEqualityTester()
