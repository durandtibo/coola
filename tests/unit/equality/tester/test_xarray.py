from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola.equality.config import EqualityConfig2
from coola.equality.tester import (
    XarrayDataArrayEqualityTester,
    XarrayDatasetEqualityTester,
    XarrayVariableEqualityTester,
)
from coola.testing.fixtures import xarray_available
from coola.utils.imports import is_numpy_available, is_xarray_available
from tests.unit.equality.utils import ExamplePair

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_xarray_available():
    import xarray as xr
else:
    xr = Mock()


@pytest.fixture
def config() -> EqualityConfig2:
    return EqualityConfig2()


XARRAY_DATA_ARRAY_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6)),
            expected=xr.DataArray(np.arange(6)),
        ),
        id="1d without dims",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6), dims=["z"]),
            expected=xr.DataArray(np.arange(6), dims=["z"]),
        ),
        id="1d with dims",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones((2, 3))),
            expected=xr.DataArray(np.ones((2, 3))),
        ),
        id="2d without dims",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones((2, 3)), dims=["x", "y"]),
            expected=xr.DataArray(np.ones((2, 3)), dims=["x", "y"]),
        ),
        id="2d with dims",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6)),
            expected=xr.DataArray(np.arange(6)),
        ),
        id="int dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6, dtype=float)),
            expected=xr.DataArray(np.arange(6, dtype=float)),
        ),
        id="float dtype",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6), dims=["x"], coords={"x": np.arange(6)}),
            expected=xr.DataArray(np.arange(6), dims=["x"], coords={"x": np.arange(6)}),
        ),
        id="dims and coords",
    ),
]
XARRAY_DATA_ARRAY_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones(6)),
            expected=xr.DataArray(np.zeros(6)),
            expected_message="objects have different variable:",
        ),
        id="different value",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6), dims=["z1"]),
            expected=xr.DataArray(np.arange(6), dims=["z2"]),
            expected_message="objects have different variable:",
        ),
        id="different dims",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6), name="meow"),
            expected=xr.DataArray(np.arange(6), name="bear"),
            expected_message="objects have different name:",
        ),
        id="different name",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6), attrs={"global": "meow"}),
            expected=xr.DataArray(np.arange(6), attrs={"global": "meoowww"}),
            expected_message="objects have different variable:",
        ),
        id="different attrs",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6), coords={"z": [1, 2, 3, 4, 5, 6]}),
            expected=xr.DataArray(np.arange(6), coords={"z": [10, 20, 30, 40, 50, 60]}),
            expected_message="objects have different _coords:",
        ),
        id="different coords",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.arange(6)),
            expected=np.arange(6),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]
XARRAY_DATA_ARRAY_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones((2, 3))),
            expected=xr.DataArray(np.full(shape=(2, 3), fill_value=1.5)),
            atol=1.0,
        ),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones((2, 3))),
            expected=xr.DataArray(np.full(shape=(2, 3), fill_value=1.05)),
            atol=0.1,
        ),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones((2, 3))),
            expected=xr.DataArray(np.full(shape=(2, 3), fill_value=1.005)),
            atol=0.01,
        ),
        id="atol=0.01",
    ),
    # rtol
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones((2, 3))),
            expected=xr.DataArray(np.full(shape=(2, 3), fill_value=1.5)),
            rtol=1.0,
        ),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones((2, 3))),
            expected=xr.DataArray(np.full(shape=(2, 3), fill_value=1.05)),
            rtol=0.1,
        ),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.DataArray(np.ones((2, 3))),
            expected=xr.DataArray(np.full(shape=(2, 3), fill_value=1.005)),
            rtol=0.01,
        ),
        id="rtol=0.01",
    ),
]

XARRAY_DATASET_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6))}),
            expected=xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6))}),
        ),
        id="data_vars",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(coords={"z": np.arange(6)}),
            expected=xr.Dataset(coords={"z": np.arange(6)}),
        ),
        id="coords",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(attrs={"global": "meow"}),
            expected=xr.Dataset(attrs={"global": "meow"}),
        ),
        id="attrs",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(
                data_vars={"x": xr.DataArray(np.arange(6))},
                coords={"z": np.arange(6)},
                attrs={"global": "meow"},
            ),
            expected=xr.Dataset(
                data_vars={"x": xr.DataArray(np.arange(6))},
                coords={"z": np.arange(6)},
                attrs={"global": "meow"},
            ),
        ),
        id="data_vars and coords and attrs",
    ),
]
XARRAY_DATASET_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.zeros(6))}),
            expected=xr.Dataset(data_vars={"x": xr.DataArray(np.ones(6))}),
            expected_message="objects have different data_vars:",
        ),
        id="different data_vars",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(coords={"z": [1, 2, 3]}),
            expected=xr.Dataset(coords={"z": [0, 1, 2]}),
            expected_message="objects have different coords:",
        ),
        id="different coords",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(attrs={"global": "meow"}),
            expected=xr.Dataset(attrs={"global": "meowwww"}),
            expected_message="objects have different attrs:",
        ),
        id="different attrs",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6), dims=["z"])}),
            expected=np.ones((2, 3)),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]
XARRAY_DATASET_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.ones((2, 3)))}),
            expected=xr.Dataset(
                data_vars={"x": xr.DataArray(np.full(shape=(2, 3), fill_value=1.5))}
            ),
            atol=1.0,
        ),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.ones((2, 3)))}),
            expected=xr.Dataset(
                data_vars={"x": xr.DataArray(np.full(shape=(2, 3), fill_value=1.05))}
            ),
            atol=0.1,
        ),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.ones((2, 3)))}),
            expected=xr.Dataset(
                data_vars={"x": xr.DataArray(np.full(shape=(2, 3), fill_value=1.005))}
            ),
            atol=0.1,
        ),
        id="atol=0.001",
    ),
    # rtol
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.ones((2, 3)))}),
            expected=xr.Dataset(
                data_vars={"x": xr.DataArray(np.full(shape=(2, 3), fill_value=1.5))}
            ),
            rtol=1.0,
        ),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.ones((2, 3)))}),
            expected=xr.Dataset(
                data_vars={"x": xr.DataArray(np.full(shape=(2, 3), fill_value=1.05))}
            ),
            rtol=0.1,
        ),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Dataset(data_vars={"x": xr.DataArray(np.ones((2, 3)))}),
            expected=xr.Dataset(
                data_vars={"x": xr.DataArray(np.full(shape=(2, 3), fill_value=1.005))}
            ),
            rtol=0.1,
        ),
        id="rtol=0.001",
    ),
]

XARRAY_VARIABLE_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.arange(6)),
            expected=xr.Variable(dims=["z"], data=np.arange(6)),
        ),
        id="1d",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["x", "y"], data=np.ones((2, 3))),
            expected=xr.Variable(dims=["x", "y"], data=np.ones((2, 3))),
        ),
        id="2d",
    ),
]
XARRAY_VARIABLE_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.ones(6)),
            expected=xr.Variable(dims=["z"], data=np.zeros(6)),
            expected_message="objects have different data:",
        ),
        id="different data",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["x"], data=np.arange(6)),
            expected=xr.Variable(dims=["y"], data=np.arange(6)),
            expected_message="objects have different dims:",
        ),
        id="different dims",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meow"}),
            expected=xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meoowww"}),
            expected_message="objects have different attrs:",
        ),
        id="different attrs",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.ones(6)),
            expected=np.ones(6),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
]
XARRAY_VARIABLE_EQUAL_TOLERANCE = [
    # atol
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.ones(3)),
            expected=xr.Variable(dims=["z"], data=np.full(shape=3, fill_value=1.5)),
            atol=1.0,
        ),
        id="atol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.ones(3)),
            expected=xr.Variable(dims=["z"], data=np.full(shape=3, fill_value=1.05)),
            atol=0.1,
        ),
        id="atol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.ones(3)),
            expected=xr.Variable(dims=["z"], data=np.full(shape=3, fill_value=1.005)),
            atol=0.01,
        ),
        id="atol=0.01",
    ),
    # rtol
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.ones(3)),
            expected=xr.Variable(dims=["z"], data=np.full(shape=3, fill_value=1.5)),
            rtol=1.0,
        ),
        id="rtol=1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.ones(3)),
            expected=xr.Variable(dims=["z"], data=np.full(shape=3, fill_value=1.05)),
            rtol=0.1,
        ),
        id="rtol=0.1",
    ),
    pytest.param(
        ExamplePair(
            actual=xr.Variable(dims=["z"], data=np.ones(3)),
            expected=xr.Variable(dims=["z"], data=np.full(shape=3, fill_value=1.005)),
            rtol=0.01,
        ),
        id="rtol=0.01",
    ),
]

XARRAY_EQUAL = XARRAY_DATA_ARRAY_EQUAL + XARRAY_DATASET_EQUAL + XARRAY_VARIABLE_EQUAL
XARRAY_NOT_EQUAL = (
    XARRAY_DATA_ARRAY_NOT_EQUAL + XARRAY_DATASET_NOT_EQUAL + XARRAY_VARIABLE_NOT_EQUAL
)
XARRAY_EQUAL_TOLERANCE = (
    XARRAY_DATA_ARRAY_EQUAL_TOLERANCE
    + XARRAY_DATASET_EQUAL_TOLERANCE
    + XARRAY_VARIABLE_EQUAL_TOLERANCE
)


###################################################
#     Tests for XarrayDataArrayEqualityTester     #
###################################################


@xarray_available
def test_xarray_data_array_equality_tester_repr() -> None:
    assert repr(XarrayDataArrayEqualityTester()).startswith("XarrayDataArrayEqualityTester(")


@xarray_available
def test_xarray_data_array_equality_tester_str() -> None:
    assert str(XarrayDataArrayEqualityTester()).startswith("XarrayDataArrayEqualityTester(")


@xarray_available
def test_xarray_data_array_equality_tester_equal_true() -> None:
    assert XarrayDataArrayEqualityTester().equal(XarrayDataArrayEqualityTester())


@xarray_available
def test_xarray_data_array_equality_tester_equal_false_different_type() -> None:
    assert not XarrayDataArrayEqualityTester().equal(123)


@xarray_available
def test_xarray_data_array_equality_tester_equal_false_different_type_child() -> None:
    class Child(XarrayDataArrayEqualityTester): ...

    assert not XarrayDataArrayEqualityTester().equal(Child())


@xarray_available
def test_xarray_data_array_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig2,
) -> None:
    obj = xr.DataArray(np.arange(6))
    assert XarrayDataArrayEqualityTester().objects_are_equal(obj, obj, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_EQUAL)
def test_xarray_data_array_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = XarrayDataArrayEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_EQUAL)
def test_xarray_data_array_equality_tester_objects_are_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = XarrayDataArrayEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_NOT_EQUAL)
def test_xarray_data_array_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = XarrayDataArrayEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_NOT_EQUAL)
def test_xarray_data_array_equality_tester_objects_are_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = XarrayDataArrayEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert caplog.messages[-1].startswith(example.expected_message)


@xarray_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_xarray_data_array_equality_tester_objects_are_equal_nan(
    config: EqualityConfig2, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        XarrayDataArrayEqualityTester().objects_are_equal(
            xr.DataArray(np.array([0.0, float("nan"), 2.0])),
            xr.DataArray(np.array([0.0, float("nan"), 2.0])),
            config,
        )
        == equal_nan
    )


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATA_ARRAY_EQUAL_TOLERANCE)
def test_xarray_data_array_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig2
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert XarrayDataArrayEqualityTester().objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@xarray_available
def test_xarray_data_array_equality_tester_no_xarray() -> None:
    with (
        patch("coola.utils.imports.is_xarray_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'xarray' package is required but not installed."),
    ):
        XarrayDataArrayEqualityTester()


#################################################
#     Tests for XarrayDatasetEqualityTester     #
#################################################


@xarray_available
def test_xarray_dataset_equality_tester_repr() -> None:
    assert repr(XarrayDatasetEqualityTester()).startswith("XarrayDatasetEqualityTester(")


@xarray_available
def test_xarray_dataset_equality_tester_str() -> None:
    assert str(XarrayDatasetEqualityTester()).startswith("XarrayDatasetEqualityTester(")


@xarray_available
def test_xarray_dataset_equality_tester_equal_true() -> None:
    assert XarrayDatasetEqualityTester().equal(XarrayDatasetEqualityTester())


@xarray_available
def test_xarray_dataset_equality_tester_equal_false_different_type() -> None:
    assert not XarrayDatasetEqualityTester().equal(123)


@xarray_available
def test_xarray_dataset_equality_tester_equal_false_different_type_child() -> None:
    class Child(XarrayDatasetEqualityTester): ...

    assert not XarrayDatasetEqualityTester().equal(Child())


@xarray_available
def test_xarray_dataset_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig2,
) -> None:
    obj = xr.Dataset(data_vars={"x": xr.DataArray(np.arange(6), dims=["z"])})
    assert XarrayDatasetEqualityTester().objects_are_equal(obj, obj, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_EQUAL)
def test_xarray_dataset_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = XarrayDatasetEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_EQUAL)
def test_xarray_dataset_equality_tester_objects_are_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = XarrayDatasetEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_NOT_EQUAL)
def test_xarray_dataset_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = XarrayDatasetEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_NOT_EQUAL)
def test_xarray_dataset_equality_tester_objects_are_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = XarrayDatasetEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert caplog.messages[-1].startswith(example.expected_message)


@xarray_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_xarray_dataset_equality_tester_objects_are_equal_nan(
    config: EqualityConfig2, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        XarrayDatasetEqualityTester().objects_are_equal(
            xr.Dataset(data_vars={"x": xr.DataArray(np.array([0.0, float("nan"), 2.0]))}),
            xr.Dataset(data_vars={"x": xr.DataArray(np.array([0.0, float("nan"), 2.0]))}),
            config,
        )
        == equal_nan
    )


@xarray_available
@pytest.mark.parametrize("example", XARRAY_DATASET_EQUAL_TOLERANCE)
def test_xarray_dataset_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig2
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert XarrayDatasetEqualityTester().objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@xarray_available
def test_xarray_dataset_equality_tester_no_xarray() -> None:
    with (
        patch("coola.utils.imports.is_xarray_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'xarray' package is required but not installed."),
    ):
        XarrayDatasetEqualityTester()


##################################################
#     Tests for XarrayVariableEqualityTester     #
##################################################


@xarray_available
def test_xarray_variable_equality_tester_repr() -> None:
    assert repr(XarrayVariableEqualityTester()).startswith("XarrayVariableEqualityTester(")


@xarray_available
def test_xarray_variable_equality_tester_str() -> None:
    assert str(XarrayVariableEqualityTester()).startswith("XarrayVariableEqualityTester(")


@xarray_available
def test_xarray_variable_equality_tester_equal_true() -> None:
    assert XarrayVariableEqualityTester().equal(XarrayVariableEqualityTester())


@xarray_available
def test_xarray_variable_equality_tester_equal_false_different_type() -> None:
    assert not XarrayVariableEqualityTester().equal(123)


@xarray_available
def test_xarray_variable_equality_tester_equal_false_different_type_child() -> None:
    class Child(XarrayVariableEqualityTester): ...

    assert not XarrayVariableEqualityTester().equal(Child())


@xarray_available
def test_xarray_variable_equality_tester_objects_are_equal_true_same_object(
    config: EqualityConfig2,
) -> None:
    obj = xr.Variable(dims=["z"], data=np.arange(6))
    assert XarrayVariableEqualityTester().objects_are_equal(obj, obj, config)


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_EQUAL)
def test_xarray_variable_equality_tester_objects_are_equal_true(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = XarrayVariableEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_EQUAL)
def test_xarray_variable_equality_tester_objects_are_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = XarrayVariableEqualityTester()
    with caplog.at_level(logging.INFO):
        assert tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_NOT_EQUAL)
def test_xarray_variable_equality_tester_objects_are_equal_false(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    tester = XarrayVariableEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert not caplog.messages


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_NOT_EQUAL)
def test_xarray_variable_equality_tester_objects_are_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig2,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    tester = XarrayVariableEqualityTester()
    with caplog.at_level(logging.INFO):
        assert not tester.objects_are_equal(
            actual=example.actual, expected=example.expected, config=config
        )
        assert caplog.messages[-1].startswith(example.expected_message)


@xarray_available
@pytest.mark.parametrize("equal_nan", [False, True])
def test_xarray_variable_equality_tester_objects_are_equal_nan(
    config: EqualityConfig2, equal_nan: bool
) -> None:
    config.equal_nan = equal_nan
    assert (
        XarrayVariableEqualityTester().objects_are_equal(
            xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
            xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
            config,
        )
        == equal_nan
    )


@xarray_available
@pytest.mark.parametrize("example", XARRAY_VARIABLE_EQUAL_TOLERANCE)
def test_xarray_variable_equality_tester_objects_are_equal_true_tolerance(
    example: ExamplePair, config: EqualityConfig2
) -> None:
    config.atol = example.atol
    config.rtol = example.rtol
    assert XarrayVariableEqualityTester().objects_are_equal(
        actual=example.actual, expected=example.expected, config=config
    )


@xarray_available
def test_xarray_variable_equality_tester_no_xarray() -> None:
    with (
        patch("coola.utils.imports.is_xarray_available", lambda: False),
        pytest.raises(RuntimeError, match=r"'xarray' package is required but not installed."),
    ):
        XarrayVariableEqualityTester()
