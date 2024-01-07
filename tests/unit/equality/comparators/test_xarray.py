from __future__ import annotations

import logging
from unittest.mock import Mock, patch

import pytest

from coola import objects_are_equal
from coola.equality import EqualityConfig
from coola.equality.comparators.xarray_ import (
    XarrayVariableEqualityComparator,
    get_type_comparator_mapping,
)
from coola.testers import EqualityTester
from coola.testing import xarray_available
from coola.utils.imports import is_numpy_available, is_xarray_available

if is_numpy_available():
    import numpy as np
else:
    np = Mock()

if is_xarray_available():
    import xarray as xr
else:
    xr = Mock()


@pytest.fixture()
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


######################################################
#     Tests for XarrayVariableEqualityComparator     #
######################################################


@xarray_available
def test_objects_are_equal_variable() -> None:
    assert objects_are_equal(
        xr.Variable(dims=["z"], data=np.arange(6)), xr.Variable(dims=["z"], data=np.arange(6))
    )


@xarray_available
def test_variable_equality_operator_str() -> None:
    assert str(XarrayVariableEqualityComparator()).startswith("XarrayVariableEqualityComparator(")


@xarray_available
def test_variable_equality_operator__eq__true() -> None:
    assert XarrayVariableEqualityComparator() == XarrayVariableEqualityComparator()


@xarray_available
def test_variable_equality_operator__eq__false() -> None:
    assert XarrayVariableEqualityComparator() != 123


@xarray_available
def test_variable_equality_operator_clone() -> None:
    op = XarrayVariableEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


@xarray_available
def test_variable_equality_operator_equal_true(config: EqualityConfig) -> None:
    assert XarrayVariableEqualityComparator().equal(
        xr.Variable(dims=["z"], data=np.arange(6)),
        xr.Variable(dims=["z"], data=np.arange(6)),
        config,
    )


@xarray_available
def test_variable_equality_operator_equal_true_same_object(config: EqualityConfig) -> None:
    obj = xr.Variable(dims=["z"], data=np.arange(6))
    assert XarrayVariableEqualityComparator().equal(obj, obj, config)


@xarray_available
def test_variable_equality_operator_equal_true_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(
            xr.Variable(dims=["z"], data=np.arange(6)),
            xr.Variable(dims=["z"], data=np.arange(6)),
            config,
        )
        assert not caplog.messages


@xarray_available
def test_variable_equality_operator_equal_false_data(config: EqualityConfig) -> None:
    assert not XarrayVariableEqualityComparator().equal(
        xr.Variable(dims=["z"], data=np.ones(6)),
        xr.Variable(dims=["z"], data=np.zeros(6)),
        config,
    )


@xarray_available
def test_variable_equality_operator_equal_false_data_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Variable(dims=["z"], data=np.ones(6)),
            xr.Variable(dims=["z"], data=np.zeros(6)),
            config,
        )
        assert caplog.messages[-1].startswith("objects have different data:")


@xarray_available
def test_variable_equality_operator_equal_false_dims(config: EqualityConfig) -> None:
    assert not XarrayVariableEqualityComparator().equal(
        xr.Variable(dims=["z"], data=np.arange(6)),
        xr.Variable(dims=["x"], data=np.arange(6)),
        config,
    )


@xarray_available
def test_variable_equality_operator_equal_false_dims_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Variable(dims=["z"], data=np.arange(6)),
            xr.Variable(dims=["x"], data=np.arange(6)),
            config,
        )
        assert caplog.messages[-1].startswith("objects have different dims:")


@xarray_available
def test_variable_equality_operator_equal_false_different_attrs(config: EqualityConfig) -> None:
    assert not XarrayVariableEqualityComparator().equal(
        xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meow"}),
        xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meoowww"}),
        config,
    )


@xarray_available
def test_variable_equality_operator_equal_false_attrs_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meow"}),
            xr.Variable(dims=["z"], data=np.arange(6), attrs={"global": "meoowww"}),
            config,
        )
        assert caplog.messages[-1].startswith("objects have different attrs:")


@xarray_available
def test_variable_equality_operator_equal_false_different_type(config: EqualityConfig) -> None:
    assert not XarrayVariableEqualityComparator().equal(
        xr.Variable(dims=["z"], data=np.arange(6)), np.arange(6), config
    )


@xarray_available
def test_variable_equality_operator_equal_false_different_type_show_difference(
    caplog: pytest.LogCaptureFixture, config: EqualityConfig
) -> None:
    config.show_difference = True
    comparator = XarrayVariableEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(
            xr.Variable(dims=["z"], data=np.arange(6)), np.arange(6), config
        )
        assert caplog.messages[0].startswith("objects have different types:")


@xarray_available
def test_variable_equality_operator_no_xarray() -> None:
    with patch(
        "coola.utils.imports.is_xarray_available", lambda *args, **kwargs: False
    ), pytest.raises(RuntimeError, match="`xarray` package is required but not installed."):
        XarrayVariableEqualityComparator()


@xarray_available
def test_variable_equality_operator_equal_equal_nan_false(config: EqualityConfig) -> None:
    assert not XarrayVariableEqualityComparator().equal(
        xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
        xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
        config,
    )


@xarray_available
def test_variable_equality_operator_equal_equal_nan_true(config: EqualityConfig) -> None:
    config.equal_nan = True
    # TODO(TIBO): update after the new version is finished  # noqa: TD003
    assert not XarrayVariableEqualityComparator().equal(
        xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
        xr.Variable(dims=["z"], data=np.array([0.0, float("nan"), 2.0])),
        config,
    )


##########################################
#     Tests for get_mapping_equality     #
##########################################


@xarray_available
def test_get_type_comparator_mapping() -> None:
    assert get_type_comparator_mapping() == {xr.Variable: XarrayVariableEqualityComparator()}


def test_get_type_comparator_mapping_no_xarray() -> None:
    with patch(
        "coola.equality.comparators.xarray_.is_xarray_available", lambda *args, **kwargs: False
    ):
        assert get_type_comparator_mapping() == {}
