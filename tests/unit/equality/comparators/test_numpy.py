from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola import objects_are_equal
from coola.equality import EqualityConfig
from coola.equality.comparators.numpy_ import (
    NumpyArrayEqualityComparator,
    get_type_comparator_mapping,
)
from coola.testers import EqualityTester
from coola.testing import numpy_available
from coola.utils.imports import is_numpy_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


##################################################
#     Tests for NumpyArrayEqualityComparator     #
##################################################


@numpy_available
def test_objects_are_equal_array() -> None:
    assert objects_are_equal(np.ones((2, 3)), np.ones((2, 3)))


@numpy_available
def test_numpy_array_equality_comparator_str() -> None:
    assert str(NumpyArrayEqualityComparator()).startswith("NumpyArrayEqualityComparator(")


@numpy_available
def test_numpy_array_equality_comparator__eq__true() -> None:
    assert NumpyArrayEqualityComparator() == NumpyArrayEqualityComparator()


@numpy_available
def test_numpy_array_equality_comparator__eq__false_different_type() -> None:
    assert NumpyArrayEqualityComparator() != 123


@numpy_available
def test_numpy_array_equality_comparator_clone() -> None:
    op = NumpyArrayEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@numpy_available
def test_numpy_array_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    array = np.ones((2, 3))
    assert NumpyArrayEqualityComparator().equal(array, array, config)


@numpy_available
def test_numpy_array_equality_comparator_equal_true(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = NumpyArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            object1=np.ones((2, 3)),
            object2=np.ones((2, 3)),
            config=config,
        )
        assert not caplog.messages


@numpy_available
def test_numpy_array_equality_comparator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = NumpyArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            object1=np.ones((2, 3)),
            object2=np.ones((2, 3)),
            config=config,
        )
        assert not caplog.messages


@numpy_available
def test_numpy_array_equality_comparator_equal_false_different_dtype(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = NumpyArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=np.ones(shape=(2, 3), dtype=float),
            object2=np.ones(shape=(2, 3), dtype=int),
            config=config,
        )
        assert not caplog.messages


@numpy_available
def test_numpy_array_equality_comparator_equal_false_different_dtype_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = NumpyArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=np.ones(shape=(2, 3), dtype=float),
            object2=np.ones(shape=(2, 3), dtype=int),
            config=config,
        )
        assert caplog.messages[0].startswith("objects have different data types:")


@numpy_available
def test_numpy_array_equality_comparator_equal_false_different_shape(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = NumpyArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=np.ones((2, 3)), object2=np.zeros((6,)), config=config)
        assert not caplog.messages


@numpy_available
def test_numpy_array_equality_comparator_equal_false_different_shape_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = NumpyArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=np.ones((2, 3)), object2=np.zeros((6,)), config=config)
        assert caplog.messages[0].startswith("objects have different shapes:")


@numpy_available
def test_numpy_array_equality_comparator_equal_false_different_value(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = NumpyArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=np.ones((2, 3)), object2=np.zeros((2, 3)), config=config
        )
        assert not caplog.messages


@numpy_available
def test_numpy_array_equality_comparator_equal_false_different_value_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = NumpyArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            object1=np.ones((2, 3)), object2=np.zeros((2, 3)), config=config
        )
        assert caplog.messages[0].startswith("numpy.ndarrays have different elements:")


@numpy_available
def test_numpy_array_equality_comparator_equal_false_different_type(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    comparator = NumpyArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=np.ones((2, 3)), object2=42, config=config)
        assert not caplog.messages


@numpy_available
def test_numpy_array_equality_comparator_equal_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = NumpyArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=np.ones((2, 3)), object2=42, config=config)
        assert caplog.messages[0].startswith("objects have different types:")


@numpy_available
def test_numpy_array_equality_comparator_equal_nan_false(config: EqualityConfig) -> None:
    assert not NumpyArrayEqualityComparator().equal(
        object1=np.array([0.0, np.nan, np.nan, 1.2]),
        object2=np.array([0.0, np.nan, np.nan, 1.2]),
        config=config,
    )


@numpy_available
def test_numpy_array_equality_comparator_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    assert NumpyArrayEqualityComparator().equal(
        object1=np.array([0.0, np.nan, np.nan, 1.2]),
        object2=np.array([0.0, np.nan, np.nan, 1.2]),
        config=config,
    )


@numpy_available
def test_numpy_array_equality_comparator_no_numpy() -> None:
    with patch(
        "coola.utils.imports.is_numpy_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`numpy` package is required but not installed."):
        NumpyArrayEqualityComparator()


#################################################
#     Tests for get_type_comparator_mapping     #
#################################################


@numpy_available
def test_get_type_comparator_mapping() -> None:
    assert get_type_comparator_mapping() == {np.ndarray: NumpyArrayEqualityComparator()}


def test_get_type_comparator_mapping_no_numpy() -> None:
    with patch(
        "coola.equality.comparators.numpy_.is_numpy_available", lambda *args, **kwargs: False
    ):
        assert get_type_comparator_mapping() == {}
