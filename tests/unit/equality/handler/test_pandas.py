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
from coola.equality.handler.pandas import (
    extension_array_equal,
    frame_equal,
    index_equal,
    series_equal,
)
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


####################################
#     Tests for missing values     #
####################################

# A null (``None``, ``pd.NA``, ``NaT``) in a non-float column is equal to a null, whatever
# ``equal_nan``: a copy of a frame is equal to the frame. A NaN is governed by ``equal_nan``.
NULL_SERIES = [
    pytest.param(pd.Series(["a", None]), id="object None"),
    pytest.param(pd.Series(["a", None], dtype="string"), id="string"),
    pytest.param(pd.Series(pd.array([1, None], dtype="Int64")), id="Int64"),
    pytest.param(pd.Series(pd.to_datetime(["2020-01-01", None])), id="datetime NaT"),
    pytest.param(pd.Series(pd.Categorical(["a", None])), id="categorical"),
]
NAN_SERIES = [
    pytest.param(pd.Series([1.0, float("nan")]), id="float64"),
    pytest.param(pd.Series(pd.array([1.0, None], dtype="Float64")), id="Float64"),
    pytest.param(pd.Series([1.0 + 0j, complex("nan")]), id="complex"),
]


@pandas_available
@pytest.mark.parametrize("series", NULL_SERIES)
def test_series_equal_null_copy_is_equal(series: pd.Series) -> None:
    assert series_equal(series, series.copy(), EqualityConfig())


@pandas_available
@pytest.mark.parametrize("series", NULL_SERIES)
def test_frame_equal_null_copy_is_equal(series: pd.Series) -> None:
    frame = series.to_frame("a")
    assert frame_equal(frame, frame.copy(), EqualityConfig())


@pandas_available
@pytest.mark.parametrize("series", NAN_SERIES)
def test_series_equal_nan_copy_is_not_equal(series: pd.Series) -> None:
    assert not series_equal(series, series.copy(), EqualityConfig())


@pandas_available
@pytest.mark.parametrize("series", NAN_SERIES)
def test_series_equal_nan_copy_is_equal_with_equal_nan(series: pd.Series) -> None:
    assert series_equal(series, series.copy(), EqualityConfig(equal_nan=True))


@pandas_available
def test_frame_equal_nan_column_next_to_null_column_is_not_equal() -> None:
    frame = pd.DataFrame({"a": ["x", None], "b": [1.0, float("nan")]})
    assert not frame_equal(frame, frame.copy(), EqualityConfig())


@pandas_available
def test_frame_equal_nulls_at_different_positions() -> None:
    assert not frame_equal(
        pd.DataFrame({"a": ["x", None]}), pd.DataFrame({"a": [None, "x"]}), EqualityConfig()
    )


@pandas_available
def test_frame_equal_non_numeric_column_stays_exact() -> None:
    assert not frame_equal(
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        pd.DataFrame({"a": [1, 2], "b": ["x", "z"]}),
        EqualityConfig(atol=1.0),
    )


@pandas_available
def test_frame_equal_null_and_value() -> None:
    assert not frame_equal(
        pd.DataFrame({"a": ["x", None]}), pd.DataFrame({"a": ["x", "y"]}), EqualityConfig()
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
def test_index_equal_null_copy_is_equal() -> None:
    index = pd.Index(["a", None])
    assert index_equal(index, index.copy(), EqualityConfig())


@pandas_available
def test_index_equal_multi_index_null_copy_is_equal() -> None:
    index = pd.MultiIndex.from_tuples([("a", 1.0), ("b", 2.0)]).set_codes([[0, -1], [0, 1]])
    assert index_equal(index, index.copy(), EqualityConfig())


@pandas_available
def test_index_equal_multi_index_nan_copy_is_not_equal() -> None:
    index = pd.MultiIndex.from_tuples([("a", 1.0), ("b", 2.0)]).set_codes([[0, 1], [0, -1]])
    assert not index_equal(index, index.copy(), EqualityConfig())


def _durations(values: list[float | None]) -> pd.Series:
    return pd.Series(pd.to_timedelta(values, unit="s")).astype("timedelta64[ns]")


@pandas_available
def test_series_equal_timedelta_atol_in_seconds() -> None:
    assert series_equal(_durations([1, 2]), _durations([1.5, 2]), EqualityConfig(atol=1.0))


@pandas_available
def test_series_equal_timedelta_outside_atol() -> None:
    assert not series_equal(_durations([1, 2]), _durations([4, 2]), EqualityConfig(atol=1.0))


@pandas_available
def test_series_equal_timedelta_exact_without_tolerance() -> None:
    assert not series_equal(_durations([1, 2]), _durations([1.5, 2]), EqualityConfig())


@pandas_available
def test_series_equal_timedelta_nat_equal_nan() -> None:
    assert series_equal(
        _durations([1, None]), _durations([1.2, None]), EqualityConfig(atol=1.0, equal_nan=True)
    )


@pandas_available
def test_series_equal_timedelta_different_dtype_with_tolerance() -> None:
    assert not series_equal(
        _durations([1, 2]), _durations([1, 2]).astype("timedelta64[s]"), EqualityConfig(atol=1.0)
    )


@pandas_available
def test_frame_equal_timedelta_atol_in_seconds() -> None:
    assert frame_equal(
        pd.DataFrame({"a": _durations([1, 2]), "b": [1, 2]}),
        pd.DataFrame({"a": _durations([1.5, 2]), "b": [1, 2]}),
        EqualityConfig(atol=1.0),
    )


@pandas_available
def test_frame_equal_timedelta_outside_atol() -> None:
    assert not frame_equal(
        pd.DataFrame({"a": _durations([1, 2]), "b": [1, 2]}),
        pd.DataFrame({"a": _durations([4, 2]), "b": [1, 2]}),
        EqualityConfig(atol=1.0),
    )


@pandas_available
def test_frame_equal_timedelta_does_not_change_the_inputs() -> None:
    df1 = pd.DataFrame({"a": _durations([1, 2])})
    frame_equal(df1, pd.DataFrame({"a": _durations([1.5, 2])}), EqualityConfig(atol=1.0))
    assert df1["a"].dtype == "timedelta64[ns]"


@pandas_available
def test_frame_equal_without_duration_column_with_tolerance() -> None:
    assert frame_equal(
        pd.DataFrame({"a": [1.0, 2.0]}),
        pd.DataFrame({"a": [1.0, 2.0001]}),
        EqualityConfig(atol=1.0),
    )


@pandas_available
@pytest.mark.parametrize("dtype", ["bool", "boolean"])
def test_series_equal_boolean_has_no_tolerance(dtype: str) -> None:
    assert not series_equal(
        pd.Series([True, False], dtype=dtype),
        pd.Series([False, False], dtype=dtype),
        EqualityConfig(atol=1.0),
    )


@pandas_available
@pytest.mark.parametrize("dtype", ["bool", "boolean"])
def test_series_equal_boolean_equal_with_tolerance(dtype: str) -> None:
    assert series_equal(
        pd.Series([True, False], dtype=dtype),
        pd.Series([True, False], dtype=dtype),
        EqualityConfig(atol=1.0),
    )


@pandas_available
def test_frame_equal_boolean_has_no_tolerance() -> None:
    assert not frame_equal(
        pd.DataFrame({"a": [True, False], "b": [1.0, 2.0]}),
        pd.DataFrame({"a": [False, False], "b": [1.0, 2.0]}),
        EqualityConfig(atol=1.0),
    )


@pandas_available
def test_frame_equal_boolean_equal_and_numeric_close() -> None:
    assert frame_equal(
        pd.DataFrame({"a": [True, False], "b": [1.0, 2.0]}),
        pd.DataFrame({"a": [True, False], "b": [1.5, 2.0]}),
        EqualityConfig(atol=1.0),
    )


@pandas_available
def test_frame_equal_nullable_boolean_has_no_tolerance() -> None:
    assert not frame_equal(
        pd.DataFrame({"a": pd.array([True, None], dtype="boolean")}),
        pd.DataFrame({"a": pd.array([True, False], dtype="boolean")}),
        EqualityConfig(atol=1.0),
    )


@pandas_available
def test_frame_equal_boolean_does_not_change_the_inputs() -> None:
    df2 = pd.DataFrame({"a": [True, False]})
    frame_equal(pd.DataFrame({"a": [True, False]}), df2, EqualityConfig(atol=1.0))
    assert df2["a"].tolist() == [True, False]


@pandas_available
def test_index_equal_boolean_has_no_tolerance() -> None:
    assert not index_equal(
        pd.Index([True, False]), pd.Index([False, False]), EqualityConfig(atol=1.0)
    )


@pandas_available
def test_extension_array_equal_boolean_has_no_tolerance() -> None:
    assert not extension_array_equal(
        pd.array([True, False], dtype="boolean"),
        pd.array([False, False], dtype="boolean"),
        EqualityConfig(atol=1.0),
    )


@pandas_available
def test_index_equal_has_no_tolerance() -> None:
    assert not index_equal(pd.Index([1.0, 2.0]), pd.Index([1.0, 2.001]), EqualityConfig(atol=0.01))


@pandas_available
def test_index_equal_different_class() -> None:
    assert not index_equal(pd.RangeIndex(2), pd.Index([0, 1]), EqualityConfig())


@pandas_available
def test_index_equal_different_names() -> None:
    assert not index_equal(pd.Index([1], name="a"), pd.Index([1], name="b"), EqualityConfig())


@pandas_available
def test_index_equal_different_dtypes() -> None:
    assert not index_equal(pd.Index([1]), pd.Index([1.0]), EqualityConfig())


@pandas_available
def test_series_equal_object_none_equals_nan() -> None:
    assert series_equal(
        pd.Series([1, None], dtype=object),
        pd.Series([1, float("nan")], dtype=object),
        EqualityConfig(),
    )


@pandas_available
def test_series_equal_nested_cells_have_no_tolerance() -> None:
    assert not series_equal(
        pd.Series([[1.0]]), pd.Series([[1.0000001]]), EqualityConfig(atol=1.0, rtol=1.0)
    )


@pandas_available
def test_series_equal_nested_cells_equal() -> None:
    assert series_equal(pd.Series([[1.0, 2.0]]), pd.Series([[1.0, 2.0]]), EqualityConfig())


@pandas_available
def test_series_equal_different_names() -> None:
    assert not series_equal(pd.Series([1], name="a"), pd.Series([1], name="b"), EqualityConfig())


@pandas_available
def test_series_equal_nan_name_equals_nan_name() -> None:
    assert series_equal(
        pd.Series([1], name=float("nan")), pd.Series([1], name=float("nan")), EqualityConfig()
    )


@pandas_available
def test_series_equal_tuple_name() -> None:
    assert series_equal(
        pd.Series([1], name=("a", "b")), pd.Series([1], name=("a", "b")), EqualityConfig()
    )


@pandas_available
def test_series_equal_different_index() -> None:
    assert not series_equal(
        pd.Series([1, 2], index=[0, 1]), pd.Series([1, 2], index=[0, 2]), EqualityConfig()
    )


@pandas_available
def test_series_equal_complex_with_tolerance() -> None:
    assert series_equal(pd.Series([1 + 1j]), pd.Series([1.5 + 1j]), EqualityConfig(atol=0.6))


@pandas_available
def test_series_equal_complex_outside_tolerance() -> None:
    assert not series_equal(pd.Series([1 + 1j]), pd.Series([2.5 + 1j]), EqualityConfig(atol=0.6))


@pandas_available
def test_series_equal_duration_nat_at_the_same_position() -> None:
    durations = pd.to_timedelta([1, None], unit="s")
    assert series_equal(pd.Series(durations), pd.Series(durations.copy()), EqualityConfig())


@pandas_available
def test_series_equal_duration_nat_at_different_positions() -> None:
    assert not series_equal(
        pd.Series(pd.to_timedelta([1, None], unit="s")),
        pd.Series(pd.to_timedelta([None, 1], unit="s")),
        EqualityConfig(atol=5.0),
    )


@pandas_available
def test_series_equal_duration_different_resolutions() -> None:
    assert not series_equal(
        pd.Series(pd.to_timedelta([1], unit="s").as_unit("s")),
        pd.Series(pd.to_timedelta([1], unit="s").as_unit("ns")),
        EqualityConfig(),
    )


@pandas_available
def test_series_equal_timezone_aware_instants_have_no_tolerance() -> None:
    assert not series_equal(
        pd.Series(pd.to_datetime(["2020-01-01"], utc=True)),
        pd.Series(pd.to_datetime(["2020-01-02"], utc=True)),
        EqualityConfig(atol=1e9),
    )


@pandas_available
def test_series_equal_nullable_float_with_tolerance() -> None:
    assert series_equal(
        pd.Series([1.0], dtype="Float64"),
        pd.Series([1.05], dtype="Float64"),
        EqualityConfig(atol=0.1),
    )


@pandas_available
def test_frame_equal_different_columns() -> None:
    assert not frame_equal(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}), EqualityConfig())


@pandas_available
def test_frame_equal_different_column_dtypes() -> None:
    assert not frame_equal(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1.0]}), EqualityConfig())


@pandas_available
def test_frame_equal_different_index() -> None:
    assert not frame_equal(
        pd.DataFrame({"a": [1, 2]}, index=[0, 1]),
        pd.DataFrame({"a": [1, 2]}, index=[0, 2]),
        EqualityConfig(),
    )


@pandas_available
def test_frame_equal_duplicated_column_names() -> None:
    df1 = pd.DataFrame([[1, 2]], columns=["a", "a"])
    assert not frame_equal(df1, pd.DataFrame([[1, 3]], columns=["a", "a"]), EqualityConfig())


@pandas_available
def test_frame_equal_mixed_columns_with_tolerance() -> None:
    assert frame_equal(
        pd.DataFrame({"a": [1.0], "b": ["x"], "c": [True]}),
        pd.DataFrame({"a": [1.05], "b": ["x"], "c": [True]}),
        EqualityConfig(atol=0.1),
    )


@pandas_available
def test_frame_equal_stops_at_the_first_different_column() -> None:
    df1 = pd.DataFrame({"a": [1.0], "b": [1.0]})
    df2 = pd.DataFrame({"a": [2.0], "b": [1.0]})
    with patch("coola.equality.handler.pandas._values_equal", return_value=False) as mock:
        frame_equal(df1, df2, EqualityConfig())
    assert mock.call_count == 1


@pandas_available
def test_frame_equal_float_nan_column_without_equal_nan() -> None:
    df = pd.DataFrame({"a": [1.0, float("nan")]})
    assert not frame_equal(df, df.copy(), EqualityConfig())


@pandas_available
def test_frame_equal_float_nan_column_with_equal_nan() -> None:
    df = pd.DataFrame({"a": [1.0, float("nan")]})
    assert frame_equal(df, df.copy(), EqualityConfig(equal_nan=True))


@pandas_available
def test_extension_array_equal_different_dtypes() -> None:
    assert not extension_array_equal(
        pd.array([1], dtype="Int64"), pd.array([1], dtype="Int32"), EqualityConfig()
    )
