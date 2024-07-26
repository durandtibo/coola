from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from coola.equality import EqualityConfig
from coola.equality.testers import EqualityTester
from coola.utils.imports import is_pyarrow_available
from tests.unit.equality.comparators.utils import ExamplePair

if is_pyarrow_available():
    import pyarrow as pa
else:
    pa = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


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
            expected_message="objects have different data types:",
        ),
        id="array different data types",
    ),
    pytest.param(
        ExamplePair(
            actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            expected=pa.array([1, 2, 3], type=pa.int64()),
            expected_message="pyarrow.Arrays have different elements:",
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
        ),
        id="table different types",
    ),
]

PYARROW_EQUAL = PYARROW_ARRAY_EQUAL + PYARROW_TABLE_EQUAL
PYARROW_NOT_EQUAL = PYARROW_ARRAY_NOT_EQUAL + PYARROW_TABLE_NOT_EQUAL
PYARROW_EQUAL_TOLERANCE = PYARROW_ARRAY_EQUAL_TOLERANCE
PYARROW_NOT_EQUAL_TOLERANCE = PYARROW_ARRAY_NOT_EQUAL_TOLERANCE
