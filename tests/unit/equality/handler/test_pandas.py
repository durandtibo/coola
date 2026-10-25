from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from coola.equality.config import EqualityConfig
from coola.equality.handler import (
    FalseHandler,
    PandasDataFrameEqualHandler,
    PandasExtensionArrayEqualHandler,
    PandasIndexEqualHandler,
    PandasSeriesEqualHandler,
    TrueHandler,
)
from coola.equality.handler.pandas import frame_equal, series_equal
from coola.testing.fixtures import pandas_available
from coola.utils.imports import is_pandas_available
from tests.unit.equality.tester.test_pandas import (
    PANDAS_DATAFRAME_EQUAL_TOLERANCE,
    PANDAS_SERIES_EQUAL_TOLERANCE,
)

if is_pandas_available():
    import pandas as pd
else:  # pragma: no cover
    pd = Mock()

if TYPE_CHECKING:
    from tests.unit.equality.utils import ExamplePair


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig()


#################################################
#     Tests for PandasDataFrameEqualHandler     #
#################################################


def test_pandas_dataframe_equal_handler_repr() -> None:
    assert repr(PandasDataFrameEqualHandler()) == "PandasDataFrameEqualHandler()"


def test_pandas_dataframe_equal_handler_repr_with_next_handler() -> None:
    assert (
        repr(PandasDataFrameEqualHandler(FalseHandler()))
        == "PandasDataFrameEqualHandler(next_handler=FalseHandler())"
    )


def test_pandas_dataframe_equal_handler_str() -> None:
    assert str(PandasDataFrameEqualHandler()) == "PandasDataFrameEqualHandler()"


def test_pandas_dataframe_equal_handler_str_with_next_handler() -> None:
    assert str(PandasDataFrameEqualHandler(FalseHandler())) == "PandasDataFrameEqualHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            PandasDataFrameEqualHandler(), PandasDataFrameEqualHandler(), id="without next handler"
        ),
        pytest.param(
            PandasDataFrameEqualHandler(FalseHandler()),
            PandasDataFrameEqualHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_pandas_dataframe_equal_handler_equal_true(
    handler1: PandasDataFrameEqualHandler, handler2: PandasDataFrameEqualHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            PandasDataFrameEqualHandler(TrueHandler()),
            PandasDataFrameEqualHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            PandasDataFrameEqualHandler(),
            PandasDataFrameEqualHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            PandasDataFrameEqualHandler(FalseHandler()),
            PandasDataFrameEqualHandler(),
            id="other next handler is none",
        ),
        pytest.param(PandasDataFrameEqualHandler(), FalseHandler(), id="different type"),
    ],
)
def test_pandas_dataframe_equal_handler_equal_false(
    handler1: PandasDataFrameEqualHandler, handler2: object
) -> None:
    assert not handler1.equal(handler2)


def test_pandas_dataframe_equal_handler_equal_false_different_type_child() -> None:
    class Child(PandasDataFrameEqualHandler): ...

    assert not PandasDataFrameEqualHandler().equal(Child())


@pandas_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (pd.DataFrame({}), pd.DataFrame({})),
        (pd.DataFrame({"col": [1, 2, 3]}), pd.DataFrame({"col": [1, 2, 3]})),
        (
            pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                }
            ),
            pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                }
            ),
        ),
    ],
)
def test_pandas_dataframe_equal_handler_handle_true(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    config: EqualityConfig,
) -> None:
    handler = PandasDataFrameEqualHandler()
    assert handler.handle(actual, expected, config)


@pandas_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (pd.DataFrame({}), pd.DataFrame({})),
        (pd.DataFrame({"col": [1, 2, 3]}), pd.DataFrame({"col": [1, 2, 3]})),
        (
            pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                }
            ),
            pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1.1, 2.2, 3.3, 4.4, 5.5],
                    "col3": ["a", "b", "c", "d", "e"],
                }
            ),
        ),
    ],
)
def test_pandas_dataframe_equal_handler_handle_true_does_not_log(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(actual, expected, config)
        assert not caplog.messages


@pandas_available
def test_pandas_dataframe_equal_handler_handle_true_show_difference(
    config: EqualityConfig,
) -> None:
    config.show_difference = True
    handler = PandasDataFrameEqualHandler()
    assert handler.handle(
        pd.DataFrame({"col": [1, 2, 3]}), pd.DataFrame({"col": [1, 2, 3]}), config
    )


@pandas_available
def test_pandas_dataframe_equal_handler_handle_true_show_difference_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    handler = PandasDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pd.DataFrame({"col": [1, 2, 3]}), pd.DataFrame({"col": [1, 2, 3]}), config)
        assert not caplog.messages


@pandas_available
def test_pandas_dataframe_equal_handler_handle_false(
    config: EqualityConfig,
) -> None:
    handler = PandasDataFrameEqualHandler()
    assert not handler.handle(pd.DataFrame({}), pd.DataFrame({"col": [1, 2, 3]}), config)


@pandas_available
def test_pandas_dataframe_equal_handler_handle_false_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pd.DataFrame({}), pd.DataFrame({"col": [1, 2, 3]}), config)
        assert not caplog.messages


@pandas_available
def test_pandas_dataframe_equal_handler_handle_false_different_column(
    config: EqualityConfig,
) -> None:
    handler = PandasDataFrameEqualHandler()
    assert not handler.handle(
        pd.DataFrame({"col1": [1, 2, 3]}), pd.DataFrame({"col2": [1, 2, 3]}), config
    )


@pandas_available
def test_pandas_dataframe_equal_handler_handle_false_different_column_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pd.DataFrame({"col1": [1, 2, 3]}), pd.DataFrame({"col2": [1, 2, 3]}), config)
        assert not caplog.messages


@pandas_available
def test_pandas_dataframe_equal_handler_handle_false_different_value(
    config: EqualityConfig,
) -> None:
    handler = PandasDataFrameEqualHandler()
    assert not handler.handle(
        pd.DataFrame({"col": [1, 2, 3]}), pd.DataFrame({"col": [1, 2, 4]}), config
    )


@pandas_available
def test_pandas_dataframe_equal_handler_handle_false_different_value_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pd.DataFrame({"col": [1, 2, 3]}), pd.DataFrame({"col": [1, 2, 4]}), config)
        assert not caplog.messages


@pandas_available
def test_pandas_dataframe_equal_handler_handle_false_different_dtype(
    config: EqualityConfig,
) -> None:
    handler = PandasDataFrameEqualHandler()
    assert not handler.handle(
        pd.DataFrame(data={"col": [1, 2, 3]}, dtype=float),
        pd.DataFrame(data={"col": [1, 2, 3]}, dtype=int),
        config,
    )


@pandas_available
def test_pandas_dataframe_equal_handler_handle_false_different_dtype_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(
            pd.DataFrame(data={"col": [1, 2, 3]}, dtype=float),
            pd.DataFrame(data={"col": [1, 2, 3]}, dtype=int),
            config,
        )
        assert not caplog.messages


@pandas_available
def test_pandas_dataframe_equal_handler_handle_false_show_difference(
    config: EqualityConfig,
) -> None:
    config.show_difference = True
    handler = PandasDataFrameEqualHandler()
    assert not handler.handle(
        pd.DataFrame({"col": [1, 2, 3]}),
        pd.DataFrame({"col": [1, 2, 4]}),
        config=config,
    )


@pandas_available
def test_pandas_dataframe_equal_handler_handle_false_show_difference_logs(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = PandasDataFrameEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(
            pd.DataFrame({"col": [1, 2, 3]}),
            pd.DataFrame({"col": [1, 2, 4]}),
            config=config,
        )
        assert caplog.messages[0].startswith("pandas.DataFrames are different:")


@pandas_available
def test_pandas_dataframe_equal_handler_handle_equal_nan_false(config: EqualityConfig) -> None:
    assert not PandasDataFrameEqualHandler().handle(
        pd.DataFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        pd.DataFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        config,
    )


@pandas_available
def test_pandas_dataframe_equal_handler_handle_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert PandasDataFrameEqualHandler().handle(
        pd.DataFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        pd.DataFrame({"col": [0.0, float("nan"), float("nan"), 1.2]}),
        config,
    )


@pandas_available
@pytest.mark.parametrize("example", PANDAS_DATAFRAME_EQUAL_TOLERANCE)
def test_pandas_dataframe_equal_handler_handle_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PandasDataFrameEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


def test_pandas_dataframe_equal_handle_set_next_handler() -> None:
    handler = PandasDataFrameEqualHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_pandas_dataframe_equal_handle_set_next_handler_none() -> None:
    handler = PandasDataFrameEqualHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_pandas_dataframe_equal_handle_set_next_handler_incorrect() -> None:
    handler = PandasDataFrameEqualHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


##############################################
#     Tests for PandasSeriesEqualHandler     #
##############################################


def test_pandas_series_equal_handler_repr() -> None:
    assert repr(PandasSeriesEqualHandler()) == "PandasSeriesEqualHandler()"


def test_pandas_series_equal_handler_repr_with_next_handler() -> None:
    assert (
        repr(PandasSeriesEqualHandler(FalseHandler()))
        == "PandasSeriesEqualHandler(next_handler=FalseHandler())"
    )


def test_pandas_series_equal_handler_str() -> None:
    assert str(PandasSeriesEqualHandler()) == "PandasSeriesEqualHandler()"


def test_pandas_series_equal_handler_str_with_next_handler() -> None:
    assert str(PandasSeriesEqualHandler(FalseHandler())) == "PandasSeriesEqualHandler()"


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            PandasSeriesEqualHandler(), PandasSeriesEqualHandler(), id="without next handler"
        ),
        pytest.param(
            PandasSeriesEqualHandler(FalseHandler()),
            PandasSeriesEqualHandler(FalseHandler()),
            id="with next handler",
        ),
    ],
)
def test_pandas_series_equal_handler_equal_true(
    handler1: PandasSeriesEqualHandler, handler2: PandasSeriesEqualHandler
) -> None:
    assert handler1.equal(handler2)


@pytest.mark.parametrize(
    ("handler1", "handler2"),
    [
        pytest.param(
            PandasSeriesEqualHandler(TrueHandler()),
            PandasSeriesEqualHandler(FalseHandler()),
            id="different next handler",
        ),
        pytest.param(
            PandasSeriesEqualHandler(),
            PandasSeriesEqualHandler(FalseHandler()),
            id="next handler is none",
        ),
        pytest.param(
            PandasSeriesEqualHandler(FalseHandler()),
            PandasSeriesEqualHandler(),
            id="other next handler is none",
        ),
        pytest.param(PandasSeriesEqualHandler(), FalseHandler(), id="different type"),
    ],
)
def test_pandas_series_equal_handler_equal_false(
    handler1: PandasSeriesEqualHandler, handler2: object
) -> None:
    assert not handler1.equal(handler2)


def test_pandas_series_equal_handler_equal_false_different_type_child() -> None:
    class Child(PandasSeriesEqualHandler): ...

    assert not PandasSeriesEqualHandler().equal(Child())


@pandas_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (pd.Series(data=[], dtype=object), pd.Series(data=[], dtype=object)),
        (pd.Series(data=[1, 2, 3]), pd.Series(data=[1, 2, 3])),
        (pd.Series(data=["a", "b", "c"]), pd.Series(data=["a", "b", "c"])),
    ],
)
def test_pandas_series_equal_handler_handle_true(
    actual: pd.Series,
    expected: pd.Series,
    config: EqualityConfig,
) -> None:
    handler = PandasSeriesEqualHandler()
    assert handler.handle(actual, expected, config)


@pandas_available
@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        (pd.Series(data=[], dtype=object), pd.Series(data=[], dtype=object)),
        (pd.Series(data=[1, 2, 3]), pd.Series(data=[1, 2, 3])),
        (pd.Series(data=["a", "b", "c"]), pd.Series(data=["a", "b", "c"])),
    ],
)
def test_pandas_series_equal_handler_handle_true_does_not_log(
    actual: pd.Series,
    expected: pd.Series,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(actual, expected, config)
        assert not caplog.messages


@pandas_available
def test_pandas_series_equal_handler_handle_true_show_difference(
    config: EqualityConfig,
) -> None:
    config.show_difference = True
    handler = PandasSeriesEqualHandler()
    assert handler.handle(pd.Series(data=[1, 2, 3]), pd.Series(data=[1, 2, 3]), config)


@pandas_available
def test_pandas_series_equal_handler_handle_true_show_difference_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pd.Series(data=[1, 2, 3]), pd.Series(data=[1, 2, 3]), config)
        assert not caplog.messages


@pandas_available
def test_pandas_series_equal_handler_handle_false_different_shape(
    config: EqualityConfig,
) -> None:
    handler = PandasSeriesEqualHandler()
    assert not handler.handle(pd.Series(data=[1, 2, 3]), pd.Series(data=[1, 2, 3, 4]), config)


@pandas_available
def test_pandas_series_equal_handler_handle_false_different_shape_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pd.Series(data=[1, 2, 3]), pd.Series(data=[1, 2, 3, 4]), config)
        assert not caplog.messages


@pandas_available
def test_pandas_series_equal_handler_handle_false_different_dtype(
    config: EqualityConfig,
) -> None:
    handler = PandasSeriesEqualHandler()
    assert not handler.handle(
        pd.Series(data=[1, 2, 3], dtype=int),
        pd.Series(data=[1, 2, 3], dtype=float),
        config,
    )


@pandas_available
def test_pandas_series_equal_handler_handle_false_different_dtype_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(
            pd.Series(data=[1, 2, 3], dtype=int),
            pd.Series(data=[1, 2, 3], dtype=float),
            config,
        )
        assert not caplog.messages


@pandas_available
def test_pandas_series_equal_handler_handle_false_different_value(
    config: EqualityConfig,
) -> None:
    handler = PandasSeriesEqualHandler()
    assert not handler.handle(pd.Series(data=[1, 2, 3]), pd.Series(data=[1, 2, 4]), config)


@pandas_available
def test_pandas_series_equal_handler_handle_false_different_value_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(pd.Series(data=[1, 2, 3]), pd.Series(data=[1, 2, 4]), config)
        assert not caplog.messages


@pandas_available
def test_pandas_series_equal_handler_handle_false_different_index(
    config: EqualityConfig,
) -> None:
    handler = PandasSeriesEqualHandler()
    assert not handler.handle(
        pd.Series(data=[1, 2, 3]),
        pd.Series(data=[1, 2, 3], index=pd.Index([2, 3, 4])),
        config,
    )


@pandas_available
def test_pandas_series_equal_handler_handle_false_different_index_does_not_log(
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(
            pd.Series(data=[1, 2, 3]),
            pd.Series(data=[1, 2, 3], index=pd.Index([2, 3, 4])),
            config,
        )
        assert not caplog.messages


@pandas_available
def test_pandas_series_equal_handler_handle_false_show_difference(config: EqualityConfig) -> None:
    config.show_difference = True
    handler = PandasSeriesEqualHandler()
    assert not handler.handle(
        actual=pd.Series(data=[1, 2, 3]),
        expected=pd.Series(data=[1, 2, 3, 4]),
        config=config,
    )


@pandas_available
def test_pandas_series_equal_handler_handle_false_show_difference_logs(
    config: EqualityConfig, caplog: pytest.LogCaptureFixture
) -> None:
    config.show_difference = True
    handler = PandasSeriesEqualHandler()
    with caplog.at_level(logging.INFO):
        handler.handle(
            actual=pd.Series(data=[1, 2, 3]),
            expected=pd.Series(data=[1, 2, 3, 4]),
            config=config,
        )
        assert caplog.messages[0].startswith("pandas.Series are different:")


@pandas_available
def test_pandas_series_equal_handler_handle_equal_nan_false(config: EqualityConfig) -> None:
    assert not PandasSeriesEqualHandler().handle(
        pd.Series([0.0, float("nan"), float("nan"), 1.2]),
        pd.Series([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


@pandas_available
def test_pandas_series_equal_handler_handle_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert PandasSeriesEqualHandler().handle(
        pd.Series([0.0, float("nan"), float("nan"), 1.2]),
        pd.Series([0.0, float("nan"), float("nan"), 1.2]),
        config,
    )


@pandas_available
@pytest.mark.parametrize("example", PANDAS_SERIES_EQUAL_TOLERANCE)
def test_pandas_series_equal_handler_handle_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert PandasSeriesEqualHandler().handle(
        actual=example.actual, expected=example.expected, config=config
    )


def test_pandas_series_equal_handle_set_next_handler() -> None:
    handler = PandasSeriesEqualHandler()
    handler.set_next_handler(FalseHandler())
    assert handler.next_handler.equal(FalseHandler())


def test_pandas_series_equal_handle_set_next_handler_none() -> None:
    handler = PandasSeriesEqualHandler()
    handler.set_next_handler(None)
    assert handler.next_handler is None


def test_pandas_series_equal_handle_set_next_handler_incorrect() -> None:
    handler = PandasSeriesEqualHandler()
    with pytest.raises(TypeError, match=r"Incorrect type for 'handler'."):
        handler.set_next_handler(42)


@pandas_available
def test_pandas_frame_equal_different_shape() -> None:
    config = EqualityConfig(equal_nan=True)
    assert not frame_equal(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1, 2, 3]}), config)


@pandas_available
def test_pandas_series_equal_different_shape() -> None:
    config = EqualityConfig(equal_nan=True)
    assert not series_equal(pd.Series([1, 2]), pd.Series([1, 2, 3]), config)


@pandas_available
def test_pandas_index_equal_handler_handle() -> None:
    handler = PandasIndexEqualHandler()
    assert handler.handle(pd.Index([1, 2]), pd.Index([1, 2]), EqualityConfig())


@pandas_available
def test_pandas_index_equal_handler_handle_false() -> None:
    handler = PandasIndexEqualHandler()
    assert not handler.handle(pd.Index([1, 2]), pd.Index([1, 3]), EqualityConfig())


@pandas_available
def test_pandas_extension_array_equal_handler_handle() -> None:
    handler = PandasExtensionArrayEqualHandler()
    assert handler.handle(pd.Categorical([1, 2]), pd.Categorical([1, 2]), EqualityConfig())


@pandas_available
def test_pandas_extension_array_equal_handler_handle_false() -> None:
    handler = PandasExtensionArrayEqualHandler()
    assert not handler.handle(pd.Categorical([1, 2]), pd.Categorical([1, 3]), EqualityConfig())


#############################################
#     Tests for the tolerance of numbers    #
#############################################


@pandas_available
def test_series_equal_integers_rtol() -> None:
    assert series_equal(pd.Series([100]), pd.Series([101]), EqualityConfig(rtol=0.01))


@pandas_available
def test_series_equal_integers_outside_rtol() -> None:
    assert not series_equal(pd.Series([100]), pd.Series([110]), EqualityConfig(rtol=0.01))


@pandas_available
def test_series_equal_atol_and_rtol_add_up() -> None:
    # |0 - 0.9| = 0.9 <= 0.5 + 0.5 * 0.9
    assert series_equal(pd.Series([0.0]), pd.Series([0.9]), EqualityConfig(atol=0.5, rtol=0.5))


@pandas_available
def test_series_equal_unsigned_integers_do_not_wrap_around() -> None:
    assert not series_equal(
        pd.Series([0], dtype="uint8"), pd.Series([200], dtype="uint8"), EqualityConfig(atol=1.0)
    )


@pandas_available
def test_series_equal_numeric_different_index() -> None:
    assert not series_equal(
        pd.Series([1.0], index=["a"]), pd.Series([1.0], index=["b"]), EqualityConfig(atol=1.0)
    )


@pandas_available
def test_series_equal_numeric_different_name() -> None:
    assert not series_equal(
        pd.Series([1.0], name="a"), pd.Series([1.0], name="b"), EqualityConfig(atol=1.0)
    )


@pandas_available
def test_series_equal_numeric_different_dtype() -> None:
    assert not series_equal(
        pd.Series([1.0], dtype="float64"),
        pd.Series([1.0], dtype="float32"),
        EqualityConfig(atol=1.0),
    )


@pandas_available
def test_series_equal_nullable_integers_nulls_at_the_same_positions() -> None:
    assert series_equal(
        pd.Series([1, None], dtype="Int64"),
        pd.Series([2, None], dtype="Int64"),
        EqualityConfig(atol=1.0, equal_nan=True),
    )


@pandas_available
def test_series_equal_nullable_integers_nulls_at_different_positions() -> None:
    assert not series_equal(
        pd.Series([1, None], dtype="Int64"),
        pd.Series([None, 1], dtype="Int64"),
        EqualityConfig(atol=1.0, equal_nan=True),
    )


@pandas_available
def test_series_equal_nullable_floats_with_equal_nan() -> None:
    assert series_equal(
        pd.Series([1.0, None], dtype="Float64"),
        pd.Series([1.5, None], dtype="Float64"),
        EqualityConfig(atol=1.0, equal_nan=True),
    )


@pandas_available
def test_series_equal_non_numeric_stays_exact() -> None:
    assert not series_equal(pd.Series(["a"]), pd.Series(["b"]), EqualityConfig(atol=1.0))


@pandas_available
def test_frame_equal_integers_atol() -> None:
    assert frame_equal(
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        pd.DataFrame({"a": [1, 3], "b": ["x", "y"]}),
        EqualityConfig(atol=1.0),
    )


@pandas_available
def test_frame_equal_integers_outside_atol() -> None:
    assert not frame_equal(
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        pd.DataFrame({"a": [1, 5], "b": ["x", "y"]}),
        EqualityConfig(atol=1.0),
    )


@pandas_available
def test_frame_equal_non_numeric_column_stays_exact() -> None:
    assert not frame_equal(
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        pd.DataFrame({"a": [1, 2], "b": ["x", "z"]}),
        EqualityConfig(atol=1.0),
    )


@pandas_available
def test_frame_equal_numeric_different_index() -> None:
    assert not frame_equal(
        pd.DataFrame({"a": [1, 2]}),
        pd.DataFrame({"a": [1, 2]}, index=[5, 6]),
        EqualityConfig(atol=1.0),
    )


@pandas_available
def test_frame_equal_numeric_different_column_names() -> None:
    assert not frame_equal(
        pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"b": [1, 2]}), EqualityConfig(atol=1.0)
    )


@pandas_available
def test_frame_equal_numeric_different_dtypes() -> None:
    assert not frame_equal(
        pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1.0, 2.0]}), EqualityConfig(atol=1.0)
    )


@pandas_available
def test_frame_equal_does_not_change_the_inputs() -> None:
    df1, df2 = pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1, 3]})
    frame_equal(df1, df2, EqualityConfig(atol=1.0))
    assert df2["a"].tolist() == [1, 3]


@pandas_available
def test_frame_equal_without_numpy_uses_the_pandas_tolerance() -> None:
    with patch("coola.equality.handler.pandas.is_numpy_available", lambda: False):
        assert frame_equal(
            pd.DataFrame({"a": [100.0]}), pd.DataFrame({"a": [100.5]}), EqualityConfig(rtol=0.01)
        )
