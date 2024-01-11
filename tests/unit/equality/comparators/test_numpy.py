from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola.equality import EqualityConfig
from coola.equality.comparators.numpy_ import (
    NumpyArrayEqualityComparator,
    NumpyMaskedArrayEqualityComparator,
    get_type_comparator_mapping,
)
from coola.equality.testers import EqualityTester
from coola.testing import numpy_available
from coola.utils.imports import is_numpy_available
from tests.unit.equality.comparators.utils import ExamplePair
from tests.unit.equality.handlers.test_numpy import NUMPY_ARRAY_TOLERANCE

if is_numpy_available():
    import numpy as np
else:
    np = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


NUMPY_ARRAY_EQUAL = [
    pytest.param(
        ExamplePair(
            object1=np.ones(shape=(2, 3), dtype=float), object2=np.ones(shape=(2, 3), dtype=float)
        ),
        id="float dtype",
    ),
    pytest.param(
        ExamplePair(
            object1=np.ones(shape=(2, 3), dtype=int), object2=np.ones(shape=(2, 3), dtype=int)
        ),
        id="int dtype",
    ),
    pytest.param(ExamplePair(object1=np.ones(shape=6), object2=np.ones(shape=6)), id="1d array"),
    pytest.param(
        ExamplePair(object1=np.ones(shape=(2, 3)), object2=np.ones(shape=(2, 3))), id="2d array"
    ),
]


NUMPY_ARRAY_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            object1=np.ones(shape=(2, 3), dtype=float),
            object2=np.ones(shape=(2, 3), dtype=int),
            expected_message="objects have different data types:",
        ),
        id="different data types",
    ),
    pytest.param(
        ExamplePair(
            object1=np.ones(shape=(2, 3)),
            object2=np.ones(shape=6),
            expected_message="objects have different shapes:",
        ),
        id="different shapes",
    ),
    pytest.param(
        ExamplePair(
            object1=np.ones(shape=(2, 3)),
            object2=np.zeros(shape=(2, 3)),
            expected_message="numpy.ndarrays have different elements:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            object1=np.ones(shape=(2, 3)),
            object2="meow",
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]


NUMPY_MASKED_ARRAY_EQUAL = [
    pytest.param(
        ExamplePair(
            object1=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0], dtype=float),
            object2=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0], dtype=float),
        ),
        id="float dtype",
    ),
    pytest.param(
        ExamplePair(
            object1=np.ma.array(data=[0, 1, 2], mask=[0, 1, 0], dtype=int),
            object2=np.ma.array(data=[0, 1, 2], mask=[0, 1, 0], dtype=int),
        ),
        id="int dtype",
    ),
    pytest.param(
        ExamplePair(
            object1=np.ma.array(data=[0.0, 1.0, 1.2]), object2=np.ma.array(data=[0.0, 1.0, 1.2])
        ),
        id="1d array",
    ),
    pytest.param(
        ExamplePair(
            object1=np.ma.array(data=np.ones(shape=(2, 3)), mask=[[0, 1, 0], [1, 0, 0]]),
            object2=np.ma.array(data=np.ones(shape=(2, 3)), mask=[[0, 1, 0], [1, 0, 0]]),
        ),
        id="2d array",
    ),
]

NUMPY_MASKED_ARRAY_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            object1=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0], dtype=float),
            object2=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0], dtype=int),
            expected_message="objects have different data types:",
        ),
        id="different data types",
    ),
    pytest.param(
        ExamplePair(
            object1=np.ma.array(data=np.ones(shape=(2, 3)), mask=[[0, 1, 0], [1, 0, 0]]),
            object2=np.ma.array(data=np.ones(shape=(3, 2)), mask=[[0, 1], [1, 0], [0, 0]]),
            expected_message="objects have different shapes:",
        ),
        id="different shapes",
    ),
    pytest.param(
        ExamplePair(
            object1=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0]),
            object2=np.ma.array(data=[0.0, 1.0, 2.0], mask=[0, 1, 0]),
            expected_message="objects have different data:",
        ),
        id="different values",
    ),
    pytest.param(
        ExamplePair(
            object1=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0]),
            object2=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 0, 1]),
            expected_message="objects have different mask:",
        ),
        id="different mask",
    ),
    pytest.param(
        ExamplePair(
            object1=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0], fill_value=-1),
            object2=np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0], fill_value=42),
            expected_message="objects have different fill_value:",
        ),
        id="different fill_value",
    ),
    pytest.param(
        ExamplePair(
            object1=np.ma.array(data=np.ones(shape=(2, 3)), mask=[[0, 1, 0], [1, 0, 0]]),
            object2=np.ones(shape=(2, 3)),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]


NUMPY_EQUAL = NUMPY_ARRAY_EQUAL + NUMPY_MASKED_ARRAY_EQUAL
NUMPY_NOT_EQUAL = NUMPY_ARRAY_NOT_EQUAL + NUMPY_MASKED_ARRAY_NOT_EQUAL


##################################################
#     Tests for NumpyArrayEqualityComparator     #
##################################################


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
@pytest.mark.parametrize("example", NUMPY_ARRAY_EQUAL)
def test_numpy_array_equality_comparator_equal_yes(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = NumpyArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_EQUAL)
def test_numpy_array_equality_comparator_equal_yes_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = NumpyArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_NOT_EQUAL)
def test_numpy_array_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = NumpyArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_NOT_EQUAL)
def test_numpy_array_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = NumpyArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@numpy_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_numpy_array_equality_comparator_equal_nan_true(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        NumpyArrayEqualityComparator().equal(
            object1=np.array([0.0, np.nan, np.nan, 1.2]),
            object2=np.array([0.0, np.nan, np.nan, 1.2]),
            config=config,
        )
        == equal_nan
    )


@numpy_available
@pytest.mark.parametrize("example", NUMPY_ARRAY_TOLERANCE)
def test_numpy_array_equality_comparator_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert NumpyArrayEqualityComparator().equal(
        object1=example.object1, object2=example.object2, config=config
    )


@numpy_available
def test_numpy_array_equality_comparator_no_numpy() -> None:
    with patch(
        "coola.utils.imports.is_numpy_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`numpy` package is required but not installed."):
        NumpyArrayEqualityComparator()


########################################################
#     Tests for NumpyMaskedArrayEqualityComparator     #
########################################################


@numpy_available
def test_numpy_masked_array_equality_comparator_str() -> None:
    assert str(NumpyMaskedArrayEqualityComparator()).startswith(
        "NumpyMaskedArrayEqualityComparator("
    )


@numpy_available
def test_numpy_masked_array_equality_comparator__eq__true() -> None:
    assert NumpyMaskedArrayEqualityComparator() == NumpyMaskedArrayEqualityComparator()


@numpy_available
def test_numpy_masked_array_equality_comparator__eq__false_different_type() -> None:
    assert NumpyMaskedArrayEqualityComparator() != 123


@numpy_available
def test_numpy_masked_array_equality_comparator_clone() -> None:
    op = NumpyMaskedArrayEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@numpy_available
def test_numpy_masked_array_equality_comparator_equal_true_same_object(
    config: EqualityConfig,
) -> None:
    array = np.ma.array(data=[0.0, 1.0, 1.2], mask=[0, 1, 0])
    assert NumpyMaskedArrayEqualityComparator().equal(array, array, config)


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_EQUAL)
def test_numpy_masked_array_equality_comparator_equal_yes(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = NumpyMaskedArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_EQUAL)
def test_numpy_masked_array_equality_comparator_equal_yes_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = NumpyMaskedArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_NOT_EQUAL)
def test_numpy_masked_array_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = NumpyMaskedArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert not caplog.messages


@numpy_available
@pytest.mark.parametrize("example", NUMPY_MASKED_ARRAY_NOT_EQUAL)
def test_numpy_masked_array_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = NumpyMaskedArrayEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(object1=example.object1, object2=example.object2, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@numpy_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_numpy_masked_array_equality_comparator_equal_nan(
    config: EqualityConfig, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        NumpyMaskedArrayEqualityComparator().equal(
            object1=np.ma.array(data=[0.0, np.nan, np.nan, 1.2], mask=[0, 1, 0, 1]),
            object2=np.ma.array(data=[0.0, np.nan, np.nan, 1.2], mask=[0, 1, 0, 1]),
            config=config,
        )
        == equal_nan
    )


@numpy_available
def test_numpy_masked_array_equality_comparator_no_numpy() -> None:
    with patch(
        "coola.utils.imports.is_numpy_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`numpy` package is required but not installed."):
        NumpyMaskedArrayEqualityComparator()


#################################################
#     Tests for get_type_comparator_mapping     #
#################################################


@numpy_available
def test_get_type_comparator_mapping() -> None:
    assert get_type_comparator_mapping() == {
        np.ndarray: NumpyArrayEqualityComparator(),
        np.ma.MaskedArray: NumpyMaskedArrayEqualityComparator(),
    }


def test_get_type_comparator_mapping_no_numpy() -> None:
    with patch(
        "coola.equality.comparators.numpy_.is_numpy_available", lambda *args, **kwargs: False
    ):
        assert get_type_comparator_mapping() == {}
