from __future__ import annotations

import logging
import warnings
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.tester import (
    PyarrowEqualityTester,
)
from coola.testing.fixtures import pyarrow_available
from coola.utils.imports import is_pyarrow_available
from tests.unit.equality.utils import ExamplePair

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
                    datetime(year=2012, month=1, day=1, tzinfo=timezone.utc),
                    datetime(year=2012, month=3, day=4, tzinfo=timezone.utc),
                ],
                type=pa.date64(),
            ),
            expected=pa.array(
                [
                    datetime(year=2012, month=1, day=1, tzinfo=timezone.utc),
                    datetime(year=2012, month=3, day=4, tzinfo=timezone.utc),
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
                    datetime(year=2012, month=1, day=1, tzinfo=timezone.utc),
                    datetime(year=2012, month=3, day=4, tzinfo=timezone.utc),
                ],
                type=pa.date64(),
            ),
            expected=pa.array(
                [
                    datetime(year=2012, month=1, day=1, tzinfo=timezone.utc),
                    datetime(year=2012, month=3, day=4, tzinfo=timezone.utc),
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
                    datetime(year=2012, month=1, day=1, tzinfo=timezone.utc),
                    datetime(year=2012, month=3, day=4, tzinfo=timezone.utc),
                ],
                type=pa.date64(),
            ),
            expected=pa.array(
                [
                    datetime(year=2012, month=1, day=1, tzinfo=timezone.utc),
                    datetime(year=2012, month=3, day=4, tzinfo=timezone.utc),
                ],
                type=pa.date64(),
            ),
            rtol=0.01,
        ),
        id="array date64 rtol",
    ),
]
PYARROW_ARRAY_NOT_EQUAL_TOLERANCE = [
    # The arrays are not equal because the atol and rtol arguments are ignored.
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
PYARROW_EQUAL_TOLERANCE = PYARROW_ARRAY_EQUAL_TOLERANCE
PYARROW_NOT_EQUAL_TOLERANCE = PYARROW_ARRAY_NOT_EQUAL_TOLERANCE


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
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = PyarrowEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_EQUAL)
def test_pyarrow_equality_tester_objects_are_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = PyarrowEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_NOT_EQUAL)
def test_pyarrow_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = PyarrowEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_NOT_EQUAL)
def test_pyarrow_equality_tester_objects_are_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = PyarrowEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert caplog.messages[-1].startswith(example.expected_message)


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
    with warnings.catch_warnings(record=True) as w:
        # Not equal because equal_nan is ignored
        assert not PyarrowEqualityTester().objects_are_equal(
            actual=pa.array([0.0, float("nan"), float("nan"), 1.2]),
            expected=pa.array([0.0, float("nan"), float("nan"), 1.2]),
            config=config,
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "equal_nan is ignored because it is not supported" in str(w[-1].message)


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_EQUAL_TOLERANCE)
def test_pyarrow_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    with warnings.catch_warnings(record=True) as w:
        assert PyarrowEqualityTester().objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "tol is ignored because it is not supported" in str(w[-1].message)


def test_pyarrow_equality_tester_no_pyarrow() -> None:
    with (
        patch("coola.utils.imports.is_pyarrow_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'pyarrow' package is required but not installed."),
    ):
        PyarrowEqualityTester()
