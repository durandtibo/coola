from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from coola.equality import objects_are_allclose
from coola.equality.config import EqualityConfig
from coola.equality.handler import FalseHandler, PyarrowEqualHandler, TrueHandler
from coola.equality.handler.pyarrow import object_equal
from coola.testing.fixtures import pyarrow_available
from coola.utils.imports import is_pyarrow_available
from tests.unit.equality.tester.test_pyarrow import (
    PYARROW_EQUAL,
    PYARROW_EQUAL_TOLERANCE,
    PYARROW_NOT_EQUAL,
    PYARROW_NOT_EQUAL_TOLERANCE,
)
from tests.unit.equality.utils import messages

if TYPE_CHECKING:
    from tests.unit.equality.utils import ExamplePair

if is_pyarrow_available():
    import pyarrow as pa
else:
    pa = Mock()


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


#########################################
#     Tests for PyarrowEqualHandler     #
#########################################


def test_pyarrow_equal_handler_repr() -> None:
    assert repr(PyarrowEqualHandler()) == "PyarrowEqualHandler()"


def test_pyarrow_equal_handler_repr_with_next_handler() -> None:
    assert (
        repr(PyarrowEqualHandler(FalseHandler()))
        == "PyarrowEqualHandler(next_handler=FalseHandler())"
    )


def test_pyarrow_equal_handler_str() -> None:
    assert str(PyarrowEqualHandler()) == "PyarrowEqualHandler()"


def test_pyarrow_equal_handler_str_with_next_handler() -> None:
    assert str(PyarrowEqualHandler(FalseHandler())) == "PyarrowEqualHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(PyarrowEqualHandler(), PyarrowEqualHandler(), id="without next handler"),
        pytest.param(
            PyarrowEqualHandler(FalseHandler()),
            PyarrowEqualHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_pyarrow_equal_handler_equal_true(
    handler1: PyarrowEqualHandler, handler2: PyarrowEqualHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            PyarrowEqualHandler(TrueHandler()),
            PyarrowEqualHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            PyarrowEqualHandler(),
            PyarrowEqualHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            PyarrowEqualHandler(FalseHandler()),
            PyarrowEqualHandler(),
            id="other next handler is none",
        ),
        pytest.param(PyarrowEqualHandler(), FalseHandler(), id="different type"),
    ],
)
def test_pyarrow_equal_handler_equal_false(handler1: PyarrowEqualHandler, handler2: object) -> None:
    assert not handler1.equal(handler2)


def test_pyarrow_equal_handler_equal_false_different_type_child() -> None:
    class Child(PyarrowEqualHandler): ...

    assert not PyarrowEqualHandler().equal(Child())


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_EQUAL)
def test_pyarrow_equal_handler_handle_true(example: ExamplePair, config: EqualityConfig) -> None:
    assert PyarrowEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_NOT_EQUAL)
def test_pyarrow_equal_handler_handle_false(example: ExamplePair, config: EqualityConfig) -> None:
    assert not PyarrowEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


@pyarrow_available
def test_pyarrow_equal_handler_handle_equal_nan_false(config: EqualityConfig) -> None:
    assert not PyarrowEqualHandler().handle(
        pa.array([0.0, float("nan"), float("nan"), 1.2]),
        pa.array([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


@pyarrow_available
def test_pyarrow_equal_handler_handle_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert PyarrowEqualHandler().handle(
        pa.array([0.0, float("nan"), float("nan"), 1.2]),
        pa.array([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


@pyarrow_available
def test_pyarrow_equal_handler_handle_false_explaining(config: EqualityConfig) -> None:
    config.reports = []
    handler = PyarrowEqualHandler()
    assert not handler.handle(
        actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        expected=pa.array([1.0, 2.0, 4.0], type=pa.float64()),
        config=config,
    )


@pyarrow_available
def test_pyarrow_equal_handler_handle_false_explaining_reports(config: EqualityConfig) -> None:
    config.reports = []
    handler = PyarrowEqualHandler()
    handler.handle(
        actual=pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        expected=pa.array([1.0, 2.0, 4.0], type=pa.float64()),
        config=config,
    )
    assert messages(config)[0].startswith("objects are different")


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_EQUAL_TOLERANCE)
def test_pyarrow_equal_handler_handle_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PyarrowEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_NOT_EQUAL_TOLERANCE)
def test_pyarrow_equal_handler_handle_false_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert not PyarrowEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


def test_pyarrow_equal_handler_set_next_handler() -> None:
    PyarrowEqualHandler().set_next_handler(FalseHandler())


def test_pyarrow_equal_handle_set_next_handler() -> None:
    handler = PyarrowEqualHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_pyarrow_equal_handle_set_next_handler_none() -> None:
    handler = PyarrowEqualHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_pyarrow_equal_handle_set_next_handler_incorrect() -> None:
    handler = PyarrowEqualHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


##################################
#     Tests for object_equal     #
##################################


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_EQUAL)
def test_object_equal_true(example: ExamplePair, config: EqualityConfig) -> None:
    assert object_equal(example.actual, example.expected, config=config)


@pyarrow_available
@pytest.mark.parametrize("example", PYARROW_NOT_EQUAL)
def test_object_equal_false(example: ExamplePair, config: EqualityConfig) -> None:
    assert not object_equal(example.actual, example.expected, config=config)


@pyarrow_available
def test_pyarrow_equal_handler_chunked_array() -> None:
    handler = PyarrowEqualHandler()
    assert handler.handle(pa.chunked_array([[1, 2]]), pa.chunked_array([[1, 2]]), EqualityConfig())


@pyarrow_available
def test_pyarrow_equal_handler_chunked_array_different() -> None:
    handler = PyarrowEqualHandler()
    assert not handler.handle(
        pa.chunked_array([[1, 2]]), pa.chunked_array([[1, 3]]), EqualityConfig()
    )


@pyarrow_available
def test_pyarrow_equal_handler_record_batch() -> None:
    handler = PyarrowEqualHandler()
    assert handler.handle(
        pa.record_batch({"a": [1]}), pa.record_batch({"a": [1]}), EqualityConfig()
    )


@pyarrow_available
def test_pyarrow_equal_handler_record_batch_different() -> None:
    handler = PyarrowEqualHandler()
    assert not handler.handle(
        pa.record_batch({"a": [1]}), pa.record_batch({"a": [2]}), EqualityConfig()
    )


@pyarrow_available
def test_object_equal_null_positions_are_the_same() -> None:
    config = EqualityConfig(atol=0.1)
    assert object_equal(pa.array([1.0, None]), pa.array([1.05, None]), config)


@pyarrow_available
def test_object_equal_null_positions_must_match() -> None:
    config = EqualityConfig(atol=0.1)
    assert not object_equal(pa.array([1.0, None]), pa.array([1.0, 0.0]), config)


@pyarrow_available
def test_object_equal_table_tolerance() -> None:
    config = EqualityConfig(atol=0.1)
    table = pa.table({"a": [1.0, 2.0], "b": ["x", "y"]})
    assert object_equal(table, pa.table({"a": [1.0, 2.05], "b": ["x", "y"]}), config)


@pyarrow_available
def test_object_equal_table_different_value() -> None:
    config = EqualityConfig(atol=0.1)
    table = pa.table({"a": [1.0, 2.0], "b": ["x", "y"]})
    assert not object_equal(table, pa.table({"a": [1.0, 2.5], "b": ["x", "y"]}), config)


@pyarrow_available
def test_object_equal_table_different_string_value() -> None:
    config = EqualityConfig(atol=0.1)
    table = pa.table({"a": [1.0, 2.0], "b": ["x", "y"]})
    assert not object_equal(table, pa.table({"a": [1.0, 2.0], "b": ["x", "z"]}), config)


@pyarrow_available
def test_object_equal_table_different_column_name() -> None:
    config = EqualityConfig(atol=0.1)
    table = pa.table({"a": [1.0, 2.0], "b": ["x", "y"]})
    assert not object_equal(table, pa.table({"a": [1.0, 2.0], "c": ["x", "y"]}), config)


@pyarrow_available
def test_object_equal_table_different_length() -> None:
    config = EqualityConfig(atol=0.1)
    table = pa.table({"a": [1.0, 2.0], "b": ["x", "y"]})
    assert not object_equal(table, pa.table({"a": [1.0], "b": ["x"]}), config)


@pyarrow_available
def test_object_equal_chunked_array_tolerance() -> None:
    config = EqualityConfig(atol=0.1)
    assert object_equal(
        pa.chunked_array([[1.0, 2.0], [3.0]]), pa.chunked_array([[1.0, 2.05, 3.0]]), config
    )


@pyarrow_available
def test_object_equal_integer_tolerance() -> None:
    assert object_equal(pa.array([1, 2]), pa.array([1, 3]), EqualityConfig(atol=1.0))


@pyarrow_available
def test_object_equal_integer_tolerance_different() -> None:
    assert not object_equal(pa.array([1, 2]), pa.array([1, 4]), EqualityConfig(atol=1.0))


@pyarrow_available
def test_object_equal_different_dtypes_with_tolerance() -> None:
    assert not object_equal(pa.array([1.0]), pa.array([1]), EqualityConfig(atol=1.0))


@pyarrow_available
def test_object_equal_array_and_chunked_array_with_tolerance() -> None:
    assert not object_equal(pa.array([1.0]), pa.chunked_array([[1.0]]), EqualityConfig(atol=1.0))


@pyarrow_available
def test_object_equal_different_lengths_with_tolerance() -> None:
    assert not object_equal(pa.array([1.0]), pa.array([1.0, 1.0]), EqualityConfig(atol=1.0))


@pyarrow_available
def test_object_equal_different_strings_with_tolerance() -> None:
    assert not object_equal(pa.array(["a"]), pa.array(["b"]), EqualityConfig(atol=1.0))


@pyarrow_available
def test_object_equal_table_and_record_batch_with_tolerance() -> None:
    assert not object_equal(
        pa.table({"a": [1.0]}), pa.record_batch({"a": [1.0]}), EqualityConfig(atol=1.0)
    )


@pyarrow_available
def test_objects_are_allclose_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert objects_are_allclose([pa.array([1.0])] * 3, [pa.array([1.05])] * 3, atol=0.1)


@pyarrow_available
def test_object_equal_scalars_with_tolerance() -> None:
    assert not object_equal(pa.scalar(1.0), pa.scalar(1.05), EqualityConfig(atol=0.1))


@pyarrow_available
@pytest.mark.parametrize("name", ["equal_nan", "atol", "rtol"])
def test_object_equal_warns_once_that_a_setting_is_ignored_without_numpy(name: str) -> None:
    value = True if name == "equal_nan" else 0.1
    config = EqualityConfig(**{name: value})
    with (
        patch("coola.equality.handler.pyarrow.is_numpy_available", return_value=False),
        pytest.warns(RuntimeWarning, match=f"{name} is ignored"),
    ):
        object_equal(pa.array([1.0]), pa.array([1.05]), config)


@pyarrow_available
@pytest.mark.parametrize("name", ["equal_nan", "atol", "rtol"])
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_object_equal_is_false_when_a_setting_is_ignored_without_numpy(name: str) -> None:
    value = True if name == "equal_nan" else 0.1
    config = EqualityConfig(**{name: value})
    with patch("coola.equality.handler.pyarrow.is_numpy_available", return_value=False):
        assert not object_equal(pa.array([1.0]), pa.array([1.05]), config)


@pyarrow_available
def test_object_equal_does_not_repeat_the_warning_without_numpy() -> None:
    config = EqualityConfig(atol=0.1)
    with patch("coola.equality.handler.pyarrow.is_numpy_available", return_value=False):
        with pytest.warns(RuntimeWarning, match="atol is ignored"):
            object_equal(pa.array([1.0]), pa.array([1.05]), config)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            object_equal(pa.array([1.0]), pa.array([1.05]), config)


@pyarrow_available
def test_object_equal_duration_with_tolerance() -> None:
    assert object_equal(
        pa.array([1_000_000], type=pa.duration("us")),
        pa.array([1_500_000], type=pa.duration("us")),
        EqualityConfig(atol=1.0),
    )


@pyarrow_available
def test_object_equal_duration_outside_tolerance() -> None:
    assert not object_equal(
        pa.array([1_000_000], type=pa.duration("us")),
        pa.array([3_000_000], type=pa.duration("us")),
        EqualityConfig(atol=1.0),
    )


@pyarrow_available
def test_object_equal_duration_nulls_at_different_positions() -> None:
    assert not object_equal(
        pa.array([1, None], type=pa.duration("s")),
        pa.array([None, 1], type=pa.duration("s")),
        EqualityConfig(atol=5.0),
    )


@pyarrow_available
def test_object_equal_duration_different_units() -> None:
    assert not object_equal(
        pa.array([1], type=pa.duration("s")),
        pa.array([1], type=pa.duration("ms")),
        EqualityConfig(),
    )


@pyarrow_available
def test_object_equal_nested_values_have_no_tolerance() -> None:
    assert not object_equal(
        pa.array([[1.0]]), pa.array([[1.0000001]]), EqualityConfig(atol=1.0, rtol=1.0)
    )


@pyarrow_available
def test_object_equal_nested_nan_is_left_to_arrow() -> None:
    assert not object_equal(
        pa.array([[float("nan")]]), pa.array([[float("nan")]]), EqualityConfig(equal_nan=True)
    )


@pyarrow_available
def test_object_equal_table_ignores_schema_metadata() -> None:
    table = pa.table({"a": [1]})
    assert object_equal(table, table.replace_schema_metadata({"k": "v"}), EqualityConfig())


@pyarrow_available
def test_object_equal_table_different_schema() -> None:
    assert not object_equal(pa.table({"a": [1]}), pa.table({"b": [1]}), EqualityConfig())


@pyarrow_available
def test_object_equal_table_different_rows() -> None:
    assert not object_equal(pa.table({"a": [1]}), pa.table({"a": [1, 2]}), EqualityConfig())


@pyarrow_available
def test_object_equal_table_stops_at_the_first_different_column() -> None:
    table1 = pa.table({"a": [1.0], "b": [1.0]})
    table2 = pa.table({"a": [2.0], "b": [1.0]})
    with patch("coola.equality.handler.pyarrow._array_equal", return_value=False) as mock:
        object_equal(table1, table2, EqualityConfig())
    assert mock.call_count == 1


@pyarrow_available
def test_object_equal_table_mixed_columns_with_tolerance() -> None:
    assert object_equal(
        pa.table({"a": [1.0], "b": ["x"]}),
        pa.table({"a": [1.05], "b": ["x"]}),
        EqualityConfig(atol=0.1),
    )


@pyarrow_available
def test_object_equal_record_batch() -> None:
    batch = pa.record_batch({"a": [1.0]})
    assert object_equal(batch, pa.record_batch({"a": [1.05]}), EqualityConfig(atol=0.1))


@pyarrow_available
def test_object_equal_chunked_array_with_different_chunks() -> None:
    assert object_equal(pa.chunked_array([[1], [2]]), pa.chunked_array([[1, 2]]), EqualityConfig())


@pyarrow_available
def test_object_equal_other_object_with_equals() -> None:
    assert object_equal(
        pa.schema([("a", pa.int64())]), pa.schema([("a", pa.int64())]), EqualityConfig()
    )


@pyarrow_available
def test_object_equal_other_object_without_equals_support() -> None:
    assert not object_equal(pa.schema([("a", pa.int64())]), 1, EqualityConfig())


@pyarrow_available
def test_object_equal_null_equals_null() -> None:
    assert object_equal(pa.array(["a", None]), pa.array(["a", None]), EqualityConfig())


@pyarrow_available
def test_object_equal_float_nan_without_equal_nan() -> None:
    assert not object_equal(pa.array([float("nan")]), pa.array([float("nan")]), EqualityConfig())
